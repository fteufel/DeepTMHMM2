# DeepTM Predictor

DeepTM is a transmembrane protein topology and membrane type predictor.

## TODO before release
- Memtype suppression for globular proteins: do we want to still predict memtype if there's an R/F in the topology?
- Clean up and fully document output types
- Make visualization less 2010s
- Handling of model weights at install time
- run test set with all memtypes and protein cases
- Clean up pyproject.toml, add license
- Check with biolib for compatibility reqs
- Pin dependency versions


## Features
- **Protein Type**: Predicts the type of protein (e.g., globular, alpha TM, beta barrel).
- **Topology Prediction**: Predicts Signal Peptides, Transmembrane Helices, Beta Barrels, Reentrant loops, and Interfacial helices.
- **Membrane Type**: Predicts the specific membrane environment across 21 different types.

## Installation

```bash
pip install .
```

## Usage

Basic usage with default checkpoints:

```bash
deeptm-predict --fasta input.fasta --output-dir results
```

### Options
- `--fasta`: Path to input FASTA file.
- `--output-dir`: Directory where results will be saved.
- `--model-dir`: (Optional) Directory containing custom `.ckpt` files. Must contain exactly 5 `topology_*.ckpt` and 5 `memtype_*.ckpt` files, otherwise the tool will abort.
- `--simplify-io`: (Flag) Use generic `i/o` (inside/outside) labels instead of membrane-specific character sets.
- `--plot`: (Flag) Generate horizontal bar plots and posterior probability tracks for each prediction.
- `--device`: Device to run inference on (defaults to `cuda` if available). We advise against using device `mps` on Apple Silicon, as we have observed numerical inconsistencies in the language model embedding that lead to wrong predictions.

## Outputs
- `predicted_topologies.3line`: Standard 3-line format.
- `TMRs.gff3`: GFF3 formatted regions with biological names.
- `predictions.json`: Detailed JSON summary of all predictions.
- `*_plot.png`: Visualizations of the prediction (only generated if `--plot` is used).

### JSON Output Format (`predictions.json`)
The `predictions.json` file contains an array of objects for each sequence:
- **`id`**: Sequence identifier from the FASTA header.
- **`type`**: Simplified structural classification (e.g., `Alpha TM`, `Beta Barrel`, `Globular + SP`).
- **`membrane_type`**: Human-readable name of the predicted membrane environment.
- **`topology`**: Predicted topology string (translated or simplified).
