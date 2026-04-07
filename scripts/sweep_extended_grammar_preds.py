"""
Sweep over (first_tm_entry, second_tm_entry) penalty values using the
extended-grammar CRF and write .preds files for each combination.

LSTM emissions are computed once per sequence per fold (all extended-grammar
models share the same neural-network weights), so the sweep only adds cheap
CRF Viterbi passes.

Output layout
-------------
<output_dir>/
  fe{first}_se{second}/fold{k}/test.preds
  original/fold{k}/test.preds          (written once, for easy comparison)

where "fe{first}" and "se{second}" use the float value with '.' replaced by 'p'
(e.g. fe-0p3_se-2p5).  None values are written as "feNone" / "seNone".

Usage (from DeepTMHMM2 root)
-----------------------------
  .venv/bin/python scripts/sweep_extended_grammar_preds.py

Optional flags
--------------
  --first-entries    comma-separated floats or "None"  (default: None,0.0,-0.3,-0.6,-1.0)
  --second-entries   comma-separated floats or "None"  (default: None,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0,-4.0)
  --training-data    path to DeepTM repo               (default: ../DeepTM)
  --checkpoint-dir   path to topology_k.ckpt files     (default: checkpoints)
  --output-dir       where to write .preds files        (default: results/ext_grammar_sweep)
  --device           cpu | mps | cuda                   (default: auto)
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from deeptmhmm2_predictor.model.lightning_deeptmhmm import DeepTMLightning
from deeptmhmm2_predictor.model.extended_grammar import apply_extended_weights
from deeptmhmm2_predictor.utils.esm_embed import get_embedding


def parse_3line(path):
    with open(path) as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    return [(lines[i], lines[i + 1], lines[i + 2]) for i in range(0, len(lines), 3)]


def setting_name(first, second):
    """Human-readable directory name for a (first_entry, second_entry) pair."""
    def fmt(v):
        if v is None:
            return 'None'
        return str(v).replace('.', 'p').replace('-', 'm')
    return f'fe{fmt(first)}_se{fmt(second)}'


def parse_entry_list(s):
    """Parse a comma-separated list of floats / 'None'."""
    result = []
    for tok in s.split(','):
        tok = tok.strip()
        result.append(None if tok == 'None' else float(tok))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-entries',  default='None,0.0,-0.3,-0.6,-1.0')
    parser.add_argument('--second-entries', default='None,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0,-4.0')
    parser.add_argument('--training-data',  default=str(Path(__file__).parent.parent.parent / 'DeepTM'))
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--output-dir',     default='results/ext_grammar_sweep')
    parser.add_argument('--device',         default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
    print(f'Using device: {args.device}')

    first_entries  = parse_entry_list(args.first_entries)
    second_entries = parse_entry_list(args.second_entries)
    grid = [(f, s) for f in first_entries for s in second_entries]
    print(f'Sweep grid: {len(grid)} settings  '
          f'(first_entries={first_entries}, second_entries={second_entries})')

    training_data = Path(args.training_data)
    out_base      = Path(args.output_dir)

    # ── load split CSV ────────────────────────────────────────────────────────
    import pandas as pd
    split_df = pd.read_csv(training_data / 'data/split_reduced/all_split.csv')
    fold_map = dict(zip(split_df['AC'], split_df['cluster']))

    # ── load all FASTA entries ────────────────────────────────────────────────
    fasta_files = [
        training_data / 'data/split_reduced/fasta/alpha.3line.fasta',
        training_data / 'data/split_reduced/fasta/beta.3line.fasta',
        training_data / 'data/split_reduced/fasta/sp.3line.fasta',
        training_data / 'data/split_reduced/fasta/tp.3line.fasta',
        training_data / 'data/split_reduced/fasta/soluble.3line.fasta',
    ]
    all_entries = []
    for f in fasta_files:
        all_entries.extend(parse_3line(str(f)))
    print(f'Loaded {len(all_entries)} sequences total')

    fold_entries = {k: [] for k in range(5)}
    for hdr, seq, lbl in all_entries:
        ac   = hdr.lstrip('>').split('|')[0]
        fold = fold_map.get(ac)
        if fold is not None:
            fold_entries[fold].append((hdr, seq, lbl))
    for k, v in fold_entries.items():
        print(f'  fold {k}: {len(v)} test sequences')

    # ── load ESM2 ─────────────────────────────────────────────────────────────
    print('\nLoading ESM2 model...')
    from esm import pretrained
    esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
    esm_model = esm_model.to(args.device).eval()

    # ── process each fold ─────────────────────────────────────────────────────
    for fold_k in range(5):
        ckpt_path = Path(args.checkpoint_dir) / f'topology_{fold_k}.ckpt'
        print(f'\n── Fold {fold_k}: {ckpt_path} ──')

        lightning  = DeepTMLightning.load_from_checkpoint(
            str(ckpt_path), strict=False, map_location='cpu', weights_only=False)
        orig_model = lightning.model.to(args.device).eval()
        orig_cfg   = orig_model.crf_config

        # Build all extended-grammar models for this fold (CRF weights differ,
        # LSTM weights are identical — emissions are computed once and reused).
        sweep_models = []
        for fe, se in grid:
            m, cfg = apply_extended_weights(orig_model, orig_cfg,
                                            first_tm_entry=fe, second_tm_entry=se)
            sweep_models.append((setting_name(fe, se), m.to(args.device).eval(), cfg))

        # Prepare output dirs and file handles
        orig_dir = out_base / 'original' / f'fold{fold_k}'
        orig_dir.mkdir(parents=True, exist_ok=True)
        orig_file = open(orig_dir / 'test.preds', 'w')

        sweep_files = {}
        for name, _, _ in sweep_models:
            d = out_base / name / f'fold{fold_k}'
            d.mkdir(parents=True, exist_ok=True)
            sweep_files[name] = open(d / 'test.preds', 'w')

        entries = fold_entries[fold_k]
        print(f'  Predicting {len(entries)} sequences × {len(grid)} settings...')

        for hdr, seq, true_lbl in tqdm(entries):
            # ESM2 embedding — expensive, done once per sequence
            emb  = get_embedding(seq, esm_model, esm_alphabet, args.device).unsqueeze(0)
            mask = torch.ones(1, len(seq), dtype=torch.bool, device=args.device)

            # LSTM emissions — computed once (shared across all extended models)
            with torch.no_grad():
                emissions = orig_model._get_network_emissions(emb, mask)  # (1, L, N_states)

            # Original grammar decode
            with torch.no_grad():
                orig_paths  = orig_model.crf.decode(emissions, mask.byte())
                orig_simple = orig_model.simplify_labels(orig_paths)
                orig_topo   = orig_cfg.get_string_from_labels(orig_simple[0])
            orig_file.write(f'{hdr}\n{seq}\n{true_lbl}\n{orig_topo}\n')

            # Extended grammar decodes — reuse emissions, only CRF differs
            for name, m, cfg in sweep_models:
                with torch.no_grad():
                    # Tile emissions to the extended state space
                    ext_emissions = cfg.tile_emissions(emissions)
                    paths  = m.crf.decode(ext_emissions, mask.byte())
                    simple = cfg.convert_crf_states_to_labels_batch(paths)
                    topo   = cfg.get_string_from_labels(simple[0])
                sweep_files[name].write(f'{hdr}\n{seq}\n{true_lbl}\n{topo}\n')

        orig_file.close()
        for f in sweep_files.values():
            f.close()

    names = ['original'] + [setting_name(fe, se) for fe, se in grid]
    print(f'\nDone.  Predictions written to {out_base}/')
    print(f'Settings: {names}')


if __name__ == '__main__':
    main()
