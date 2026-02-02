"""
Main entry point for DeepTM prediction.
Orchestrates model loading, sequence embedding generation, ensemble inference, and output formatting.
"""

import argparse
import torch
import os
import json
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from tqdm import tqdm
from esm import pretrained

from .constants import (
    TOPOLOGY_MODEL_CHECKPOINTS,
    MEMTYPE_MODEL_CHECKPOINTS,
    MEMBRANE_TYPE_NAMES,
    DISPLAY_PROT_TYPE_MAPPING,
    MEMBRANE_TYPE_THRESHOLDS
)
from .model.container import ModelContainer
from .utils.esm_embed import get_embedding
from .utils.formats import gff3_format
from .utils.plotting import plot_prediction
from .utils.postprocessing import (
    infer_structural_type, 
    suppress_implausible_predictions,
    aggregate_topology_predictions, 
    aggregate_memtype_predictions,
    translate_ab_topology,
    get_named_regions
)

def main():
    parser = argparse.ArgumentParser(description="DeepTM Prediction Tool")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model-dir", default=None, help="Directory containing model checkpoints (optional)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--simplify-io", action="store_true", help="Use generic inside/outside labels instead of membrane-specific characters")
    parser.add_argument("--plot", action="store_true", help="Generate horizontal bar plots for each prediction")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Setup checkpoints
    topo_ckpts = TOPOLOGY_MODEL_CHECKPOINTS
    mem_ckpts = MEMTYPE_MODEL_CHECKPOINTS
    
    if args.model_dir:
        model_path = Path(args.model_dir)
        found_topo = sorted([str(p) for p in model_path.glob("topology_*.ckpt")])
        found_mem = sorted([str(p) for p in model_path.glob("memtype_*.ckpt")])
        
        if len(found_topo) != 5 or len(found_mem) != 5:
            print(f"Error: Expected exactly 5 topology_*.ckpt and 5 memtype_*.ckpt files in {args.model_dir}")
            sys.exit(1)
            
        topo_ckpts = found_topo
        mem_ckpts = found_mem

    # 2. Load models
    print("Loading models...")
    models = ModelContainer(topo_ckpts, mem_ckpts, args.device)
    
    print("Loading ESM2 model...")
    esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
    esm_model.to(args.device).eval()

    # 3. Process sequences
    sequences = list(SeqIO.parse(args.fasta, "fasta"))
    print(f"Processing {len(sequences)} sequences...")
    
    all_results = []
    memtype_table_data = []
    
    three_line_path = out_dir / "predicted_topologies.3line"
    gff3_path = out_dir / "TMRs.gff3"
    
    three_line_file = open(three_line_path, "w")
    gff3_file = open(gff3_path, "w")
    gff3_file.write("##gff-version 3\n")
    
    with tqdm(total=len(sequences), desc="Predicting", unit="seq") as pbar:
        for record in sequences:
            seq_id = record.id
            sequence = str(record.seq)
            
            # Generate ESM embedding
            embedding = get_embedding(sequence, esm_model, esm_alphabet, args.device).unsqueeze(0)
            mask = torch.ones(1, len(sequence), device=args.device)
            
            # Run inference across all ensemble members
            all_topo, all_mem_probs, all_marginals = [], [], []
            for i in range(len(models.topology_models)):
                t, m_p, marg, _ = models.predict_single_model(i, embedding, mask)
                all_topo.append(t)
                all_mem_probs.append(m_p)
                all_marginals.append(marg)


            ## 
            ## Output postprocessing
            ##
            
            # Aggregate via agreement-based voting and thresholding
            final_topo_ab, final_marginals_ab = aggregate_topology_predictions(all_topo, all_marginals)
            pred_mem_indices, mem_probs = aggregate_memtype_predictions(all_mem_probs)
            final_topo_ab = suppress_implausible_predictions(final_topo_ab)

            # if we don't work in simplified I-O topo, need to map to actual sides
            if args.simplify_io:
                final_topo_out = final_topo_ab
            else:
                supported_indices = list(MEMBRANE_TYPE_THRESHOLDS.keys())
                main_mem_idx = supported_indices[np.argmax([mem_probs[i] for i in supported_indices])]
                main_mem_name = MEMBRANE_TYPE_NAMES[main_mem_idx]
                final_topo_out = translate_ab_topology(final_topo_ab, main_mem_idx)
            
            # The technical regions with biological names (renamed segments)
            named_regions = get_named_regions(final_topo_ab, main_mem_idx, args.simplify_io)
            
            # Information mapping for headers
            type_code = infer_structural_type(final_topo_ab)
            type = DISPLAY_PROT_TYPE_MAPPING[type_code]
            predicted_mem_names = [MEMBRANE_TYPE_NAMES[idx] for idx in pred_mem_indices]
            mem_names_str = "; ".join(predicted_mem_names)

            # no membrane type for non-TM proteins.
            if type_code in ["I", "S", "T"]:
                predicted_mem_names = []
                mem_names_str = ""

            # The full info for this sample - json format.
            sample_record = {
                "id": seq_id,
                "type": type,
                "membrane_types": predicted_mem_names,
                "membrane_types_probabilities": mem_probs.round(2).tolist(),
                "topology_string": final_topo_out,
                "segments": named_regions,
            }

            ##
            ## Append to output files and plot
            ##

            # 5. Save/Write results
            three_line_file.write(f">{seq_id} | {type} | {mem_names_str}\n{sequence}\n{final_topo_out}\n")
            gff3_file.write(gff3_format(seq_id, len(sequence), named_regions))
            
            if args.plot:
                # Use the descriptive type string as title
                plot_prediction(seq_id, sequence, final_topo_ab, mem_names_str, final_marginals_ab, out_dir)
            
            all_results.append(sample_record)
            
            # Prepping tabular data
            row = {"id": seq_id}
            for idx, name in MEMBRANE_TYPE_NAMES.items():
                row[name] = mem_probs.round(2)[idx]
            memtype_table_data.append(row)
            
            pbar.update(1)

    three_line_file.close()
    gff3_file.close()

    # Save final files
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    df = pd.DataFrame(memtype_table_data)
    df.to_csv(out_dir / "membrane_types.tsv", sep="\t", index=False)
        
    print(f"Predictions complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()