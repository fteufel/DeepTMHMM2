# DeepTMHMM 2

DeepTMHMM 2 is a predictor for transmembrane protein topology and membrane type. Given a protein sequence, it predicts:

- **Protein type**: Globular, Alpha-helical TM, Beta-barrel, or Transit/Signal peptide-containing variants.
- **Topology**: The arrangement of transmembrane segments, signal peptides, transit peptides, reentrant loops, and interfacial helices, annotated per residue.
- **Membrane type**: The specific membrane environment the protein resides in, across 17 types (e.g. Eukaryotic plasma membrane, Bacterial inner membrane, Mitochondrial inner membrane).

## Input

Paste or upload a protein sequence in **FASTA format**. Multiple sequences are supported.

## Options

| Option | Description |
|---|---|
| **Generate plots** | Generate a per-sequence topology plot. The plot includes a posterior probability track only if **Compute marginal probabilities** is also enabled. |
| **Simplified inside/outside labels** | Use generic `I` (inside) and `O` (outside) labels in the topology string instead of membrane-specific characters. Useful when membrane type is not of interest. |
| **Compute marginal probabilities** | Compute per-residue marginal probabilities for all topology labels. Enables the posterior probability track in plots. |

## Output

The results page shows a **per-sequence report** with the predicted protein type, membrane type(s), a topology plot, and a table of predicted segments (with start/end positions).

The following files are available for download from the results page:

| File | Description |
|---|---|
| `predictions.json` | Full JSON summary of all predictions. |
| `predicted_topologies.3line` | Three-line format: FASTA header, sequence, topology string. |
| `TMRs.gff3` | Predicted topology elements in GFF3 format. |
| `membrane_types.tsv` | Per-sequence membrane type probabilities for all 21 types. |
| `*_plot.png` | Per-sequence topology and posterior probability plot (if plots enabled). |

### JSON output fields

Each entry in `predictions.json` contains:

- **`id`**: Sequence identifier from the FASTA header.
- **`type`**: Structural classification. Possible values: `Globular`, `Globular + SP`, `Globular + TP`, `Alpha TM`, `Alpha TM + SP`, `Alpha TM + TP`, `Beta Barrel`.
- **`membrane_types`**: List of predicted membrane environment name(s).
- **`topology_string`**: Per-residue topology annotation.
- **`segments`**: List of predicted segments, each with `name`, `start`, and `end`.

## Topology labels

| Label | Meaning |
|---|---|
| `S` | Signal peptide |
| `>` | Transit peptide |
| `M` | Transmembrane helix |
| `B` | Beta-barrel strand |
| `R` | Reentrant loop |
| `F` | Interfacial helix |
| `I` / `O` | Inside / Outside (simplified mode only) |

In default mode, the two topological sides use **membrane-specific labels** — one character per side, unique to each membrane type. When **Simplified inside/outside labels** is enabled, all side residues become `I` or `O`.

## Inside and outside definitions

Each membrane type has its own biological mapping of inside and outside, and its own topology label characters. Note that "outside" is used in a topological sense when tracking proteins through the secretory pathway, which can mean what is commonly understood to be the inside intracellular organelles.

| Membrane | Inside label | Inside | Outside label | Outside |
|---|---|---|---|---|
| Archaebacterial membrane | `A` | Cytoplasmic | `a` | Extracellular |
| Bacterial Gram-negative inner membrane | `C` | Cytoplasmic | `c` | Periplasmic |
| Bacterial Gram-negative outer membrane | `d` | Extracellular | `D` | Periplasmic |
| Eukaryotic plasma membrane | `E` | Cytoplasmic | `e` | Extracellular |
| Mitochondrial inner membrane | `G` | Mitochondrial matrix | `g` | Intermembrane space |
| Endoplasmic reticulum membrane | `H` | Cytoplasmic | `h` | Lumenal |
| Thylakoid membrane | `J` | Stroma | `j` | Thylakoid space |
| Bacterial Gram-positive plasma membrane | `K` | Cytoplasmic | `k` | Periplasmic |
| Golgi membrane | `L` | Cytoplasmic | `l` | Lumenal |
| Nuclear inner membrane | `N` | Nucleus matrix | `n` | Intermembrane space |
| Endosome membrane | `P` | Cytoplasmic | `p` | Lumenal |
| Vacuole membrane | `Q` | Cytoplasmic | `q` | Lumenal |
| Vesicle membrane | `T` | Cytoplasmic | `t` | Lumenal |
| Viral membrane | `V` | Intravirion | `v` | Virion surface |
| Chloroplast outer membrane | `x` | Intermembrane space | `X` | Cytoplasmic |
| Mitochondrial outer membrane | `y` | Intermembrane space | `Y` | Cytoplasmic |
| Lysosome membrane | `3` | Cytoplasmic | `4` | Lumenal |