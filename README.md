# DeepTMHMM 2

DeepTMHMM 2 is a transmembrane protein topology and membrane type predictor.

## TODO before release
- Clean up and fully document output types
- Make visualization less 2010s
- run test set with all memtypes and protein cases
- Check with biolib for compatibility reqs


## Features
- **Protein Type**: Predicts the type of protein (e.g., globular, alpha TM, beta barrel).
- **Topology Prediction**: Predicts Signal Peptides, Transmembrane Helices, Beta Barrels, Reentrant loops, and Interfacial helices.
- **Membrane Type**: Predicts the specific membrane environment across 21 different types.

## Installation

You can either clone the repo and install or install directly from GitHub:
```bash
pip install git+https://github.com/fteufel/DeepTMHMM2.git
```

This will install the `dtm2` command line tool.

When running for the first time, DeepTMHMM 2 will fetch model weights from GitHub and cache them in `~/.cache/deeptmhmm2/`. This requires an internet connection. If you want to run DeepTMHMM 2 offline, you can download the model weights manually and place them in any directory of your choice. Then, use the `--model-dir` option to specify the path to the directory containing the model weights.

## Usage

Basic usage with default checkpoints:

```bash
dtm2 input.fasta results
```

### Options

Argument | Type  | Description |
--- | --- | ---|
`input.fasta` | `str`  |Path to input FASTA file. Must be specified. |
`results` | `str`  |Directory where results will be saved. Will be created if it doesn't exist. |
`--model-dir` | `str`  |Directory containing the `.ckpt` files. Must contain exactly 5 `topology_*.ckpt` and 5 `memtype_*.ckpt` files, otherwise the tool will abort. If not provided, defaults to the cache, downloading if necessary. |
`--simplify-io` | Flag  | Use generic `I/O` (inside/outside) labels instead of membrane-specific characters. |
`--plot` | Flag  | Generate horizontal bar plots and posterior probability tracks for each prediction. |
`--device` | `str`  | (Optional) Device to run inference on (defaults to `cuda` if available, else `cpu`). Do not use `mps` on Apple Silicon, as we have observed numerical inconsistencies in the ESM2 language model embedding that lead to wrong predictions. |


## Outputs
- `predicted_topologies.3line`: Format with 3 lines per sequence:  fasta header, amino acid sequence, and predicted topology string.
- `TMRs.gff3`: GFF3 formatted predicted topology elements.
- `predictions.json`: Detailed JSON summary of all predictions.
- `*_plot.png`: Visualizations of the prediction (only generated if `--plot` is used).

### JSON Output Format (`predictions.json`)
The `predictions.json` file contains an array of objects for each sequence:
- **`id`**: Sequence identifier from the FASTA header.
- **`type`**: Simplified structural classification (e.g., `Alpha TM`, `Beta Barrel`, `Globular + SP`).
- **`membrane_type`**: Human-readable name of the predicted membrane environment.
- **`topology`**: Predicted topology string (translated or simplified).

### Inside and outside definitions

Each membrane type has its own mapping of the two topological sides "inside" and "outside" to the context of its specific environment. The following table summarizes the mapping for each membrane type. Note that the terms "inside" and "outside" reflect the topology of the secretory pathway, so "outside" is the side that can have a signal peptide, so the assignment may be counterintuitive for intracellular organelles at first sight.

Membrane | Inside | Outside
--- | --- | ---
Archaebacterial membrane | Cytoplasmic | Extracellular
Bacterial Gram-negative inner membrane | Cytoplasmic | Periplasmic
Bacterial Gram-negative outer membrane | Extracellular | Periplasmic
Eukaryotic plasma membrane | Cytoplasmic | Extracellular
Mitochondrial inner membrane | Mitochondrial matrix | Intermembrane space
Endoplasmic reticulum membrane | Cytoplasmic | Lumenal
Thylakoid membrane | Stroma | Thylakoid space
Bacterial Gram-positive plasma membrane | Cytoplasmic | Periplasmic
Golgi membrane | Cytoplasmic | Lumenal
Nuclear inner membrane | Nucleus matrix | Intermembrane space
Endosome membrane | Cytoplasmic | Lumenal
Vacuole membrane | Cytoplasmic | Lumenal
Vesicle membrane | Cytoplasmic | Lumenal
Viral membrane | Intravirion | Virion surface
Chloroplast outer membrane | Intermembrane space | Cytoplasmic
Mitochondrial outer membrane | Intermembrane space | Cytoplasmic
Lysosome membrane | Cytoplasmic | Lumenal
