# DeepTMHMM2

DeepTMHMM2 is a transmembrane protein topology and membrane type predictor.


## Features
- **Protein Type**: Predicts the type of protein (e.g., globular, alpha TM, beta barrel).
- **Topology Prediction**: Predicts signal peptides, transmembrane helices, beta barrels, reentrant loops, and interfacial helices.
- **Membrane Type**: Predicts the specific membrane environment of the protein across 17 different types.

## Installation

You can either clone the repo and install or install directly from GitHub:
```bash
pip install git+https://github.com/fteufel/DeepTMHMM2.git
```

This will install the `dtm2` command line tool.

When running for the first time, DeepTMHMM2 will fetch model weights from GitHub and cache them in `~/.cache/deeptmhmm2/`. This requires an internet connection. If you want to run DeepTMHMM2 offline, you can download the model weights manually and place them in any directory of your choice. Then, use the `--model-dir` option to specify the path to the directory containing the model weights.

## Usage

Basic usage with fasta input:

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
`--marginals` | Flag | Compute marginal probabilities for each residue. |
`--batch-size` | `int` | (Optional, performance) Maximum number of sequences to process simultaneously (default: 96). |
`--max-length-diff` | `int` | (Optional, performance) Maximum residue length difference between the shortest and longest sequence in a single batch. We start a new batch when the length difference between the current sequence and the previous one exceeds this value to avoid excessive padding processing. (default: 500). |
`--device` | `str`  | (Optional, performance) Device to run inference on (defaults to `cuda` if available, else `cpu`). Do not use `mps` on Apple Silicon, as we have observed numerical inconsistencies in the ESM2 language model embedding that lead to wrong predictions. |


### Output files

DeepTMHMM2 processes the input sequences after sorting by length for improved batch processing performance, and writes outputs while processing. Therefore, the order of samples in the output files will differ from the input FASTA file, generally appearing from longest sequence to shortest.

- `predicted_topologies.3line`: Format with 3 lines per sequence:  fasta header, amino acid sequence, and predicted topology string.
- `TMRs.gff3`: GFF3 formatted predicted topology elements.
- `predictions.json`: Detailed JSON summary of all predictions.
- `membrane_types.tsv`: TSV formatted predicted membrane type probabilities.
- `*_plot.png`: Visualizations of the prediction (only generated if `--plot` is used).

**JSON Output Format** (`predictions.json`):

The `predictions.json` file contains an array of objects for each sequence:
- **`id`**: Sequence identifier from the FASTA header.
- **`type`**: Simplified structural classification (e.g., `Alpha TM`, `Beta Barrel`, `Globular + SP`).
- **`membrane_types`**: List of human-readable names of the predicted membrane environment(s).
- **`topology_string`**: Predicted topology string (translated or simplified).
- **`segments`**: List of predicted segments, each with `name`, `start`, and `end` fields.

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

### Python API

You can also use DeepTMHMM2 in custom Python code instead of relying on the CLI:

```python
from deeptmhmm2_predictor import DeepTMHMM2


# Load the model
model = DeepTMHMM2(device='cpu')

# Predict
# `data` can be a list or dict of amino acid sequences
# if dict, keys will be used as identifiers.
preds = model.predict(data)

# Save results
preds.to_json('predictions.json')
preds.to_3line('predicted_topologies.3line')
preds.to_gff3('TMRs.gff3')
preds.to_memtype_tsv('membrane_types.tsv')

# Results are iterable - you can access individual 
# predictions by index or id and plot them, or access 
# their properties directly.
preds[0].plot(out_dir)

print(preds[0].type)
print(preds[0].topology_string)
print(preds[0].segments)
print(preds[0].membrane_types)
print(preds[0].membrane_types_probabilities)
```
