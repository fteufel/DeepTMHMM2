"""
Python API for DeepTMHMM 2.

Usage example::

    from deeptmhmm2_predictor import DeepTMHMM2

    model = DeepTMHMM2.load()                   # downloads weights on first use
    preds = model.predict(["MAST...", "MKVL..."])

    preds.to_json("out/predictions.json")
    preds.to_3line("out/predicted_topologies.3line")
    preds.to_gff3("out/TMRs.gff3")
    preds.to_memtype_tsv("out/membrane_types.tsv")

    preds[0].plot("out/")
    simplified = preds.simplify_io()
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .constants import (
    DISPLAY_PROT_TYPE_MAPPING,
    MEMBRANE_TYPE_NAMES,
    MEMBRANE_TYPE_THRESHOLDS,
)
from .model.container import ModelContainer
from .utils.esm_embed import get_embedding
from .utils.formats import gff3_format
from .utils.plotting import plot_prediction
from .utils.postprocessing import (
    aggregate_memtype_predictions,
    aggregate_topology_predictions,
    get_named_regions,
    infer_structural_type,
    suppress_implausible_predictions,
    translate_ab_topology,
)
from .weights import ensure_weights


# ---------------------------------------------------------------------------
# Prediction — one per input sequence
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Holds the full result for a single input sequence."""

    id: str
    sequence: str
    type: str                               # e.g. "Alpha TM", "Beta Barrel"
    membrane_types: List[str]               # human-readable predicted membrane types
    membrane_types_probabilities: np.ndarray
    topology_string: str                    # translated topology (membrane-specific chars or I/O)
    segments: List[tuple]                   # [(name, start, end), ...]

    # Internal fields kept for simplify_io / re-translation
    _topology_ab: str = field(repr=False)   # raw a/b topology string
    _marginals: Optional[Dict[int, np.ndarray]] = field(repr=False, default=None)
    _main_mem_idx: int = field(repr=False, default=3)
    _simplify_io: bool = field(repr=False, default=False)

    def simplify_io(self) -> "Prediction":
        """Return a copy of this prediction with generic I/O labels instead of
        membrane-specific characters."""
        new_topo = translate_ab_topology(self._topology_ab, self._main_mem_idx, simplify_io=True)
        type_code = infer_structural_type(self._topology_ab)
        new_segments = get_named_regions(self._topology_ab, self._main_mem_idx, simplify_io=True)
        return Prediction(
            id=self.id,
            sequence=self.sequence,
            type=self.type,
            membrane_types=self.membrane_types,
            membrane_types_probabilities=self.membrane_types_probabilities,
            topology_string=new_topo,
            segments=new_segments,
            _topology_ab=self._topology_ab,
            _marginals=self._marginals,
            _main_mem_idx=self._main_mem_idx,
            _simplify_io=True,
        )

    def plot(self, output_dir: Union[str, Path]) -> None:
        """Save a topology + posterior probability plot to *output_dir*."""
        if self._marginals is None:
            print(f"Warning: no marginal probabilities available for {self.id}, skipping plot.")
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mem_names_str = "; ".join(self.membrane_types)
        plot_prediction(
            self.id,
            self.sequence,
            self._topology_ab,
            mem_names_str,
            self._marginals,
            output_dir,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "membrane_types": self.membrane_types,
            "membrane_types_probabilities": self.membrane_types_probabilities.round(2).tolist(),
            "topology_string": self.topology_string,
            "segments": [{"name": n, "start": s, "end": e} for n, s, e in self.segments],
        }


# ---------------------------------------------------------------------------
# PredictionCollection — list-like container returned by model.predict()
# ---------------------------------------------------------------------------

class PredictionCollection:
    """An ordered collection of :class:`Prediction` objects with bulk output methods."""

    def __init__(self, predictions: List[Prediction]):
        self._predictions = predictions
        self._index: Dict[str, Prediction] = {p.id: p for p in predictions}

    # --- list-like interface ------------------------------------------------

    def __len__(self) -> int:
        return len(self._predictions)

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self._predictions)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            try:
                return self._index[idx]
            except KeyError:
                raise KeyError(idx)
        result = self._predictions[idx]
        if isinstance(result, list):
            return PredictionCollection(result)
        return result

    # --- transformations ----------------------------------------------------

    def simplify_io(self) -> "PredictionCollection":
        """Return a new collection where all topology strings use generic I/O labels."""
        return PredictionCollection([p.simplify_io() for p in self._predictions])

    # --- output methods -----------------------------------------------------

    def to_json(self, path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Serialise all predictions to JSON.

        If *path* is given, write to that file and return ``None``.
        Otherwise return the JSON string.
        """
        data = [p.to_dict() for p in self._predictions]
        text = json.dumps(data, indent=4)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
            return None
        return text

    def to_3line(self, path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Write/return predictions in 3-line format (header / sequence / topology)."""
        lines = []
        for p in self._predictions:
            mem_str = "; ".join(p.membrane_types)
            lines.append(f">{p.id} | {p.type} | {mem_str}\n{p.sequence}\n{p.topology_string}")
        text = "\n".join(lines) + "\n"
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
            return None
        return text

    def to_gff3(self, path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Write/return predictions in GFF3 format."""
        header = "##gff-version 3\n"
        body = "".join(
            gff3_format(p.id, len(p.sequence), p.segments) for p in self._predictions
        )
        text = header + body
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
            return None
        return text

    def to_memtype_tsv(self, path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Write/return per-sequence membrane type probability table.

        Always returns the :class:`~pandas.DataFrame`.
        If *path* is given, also writes a TSV to that location.
        """
        rows = []
        for p in self._predictions:
            row = {"id": p.id}
            for idx, name in MEMBRANE_TYPE_NAMES.items():
                row[name] = round(float(p.membrane_types_probabilities[idx]), 2)
            rows.append(row)
        df = pd.DataFrame(rows)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, sep="\t", index=False)
        return df


# ---------------------------------------------------------------------------
# DeepTMHMM2 — the model class
# ---------------------------------------------------------------------------

class DeepTMHMM2:
    """DeepTMHMM 2 topology and membrane type predictor.

    Instantiate via the :meth:`load` classmethod::

        model = DeepTMHMM2.load()                  # auto-download weights
        model = DeepTMHMM2.load(model_dir="/path")  # custom checkpoint dir
        model = DeepTMHMM2.load(device="cuda")
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> "DeepTMHMM2":
        """Load the model ensemble and ESM2 backbone.

        Parameters
        ----------
        model_dir:
            Directory containing ``topology_*.ckpt`` and ``memtype_*.ckpt``
            files (exactly 5 of each). When ``None``, weights are fetched from
            GitHub and cached in ``~/.cache/deeptmhmm2/``.
        device:
            PyTorch device string (``"cpu"``, ``"cuda"``, …). Defaults to
            ``"cuda"`` when available, otherwise ``"cpu"``.
            Avoid ``"mps"`` on Apple Silicon — numerical inconsistencies in
            the ESM2 embeddings lead to wrong predictions.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        if model_dir is not None:
            model_path = Path(model_dir)
            topo_ckpts = sorted(str(p) for p in model_path.glob("topology_*.ckpt"))
            mem_ckpts = sorted(str(p) for p in model_path.glob("memtype_*.ckpt"))
            if len(topo_ckpts) != 5 or len(mem_ckpts) != 5:
                print(
                    f"Error: expected exactly 5 topology_*.ckpt and 5 memtype_*.ckpt "
                    f"files in {model_dir}"
                )
                sys.exit(1)
        else:
            topo_ckpts, mem_ckpts = ensure_weights()

        print("Loading DeepTMHMM 2 models...")
        self._models = ModelContainer(topo_ckpts, mem_ckpts, self._device)

        print("Loading ESM2 backbone...")
        from esm import pretrained
        self._esm_model, self._esm_alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        self._esm_model.to(self._device).eval()

    def predict(
        self,
        sequences: Union[List[str], Dict[str, str], List],
        simplify_io: bool = False,
        progress_bar: bool = True,
        compute_marginals: bool = False,
    ) -> PredictionCollection:
        """Run inference on one or more amino acid sequences.

        Parameters
        ----------
        sequences:
            Any of:

            * ``List[str]`` — raw sequences; IDs are auto-generated (``seq_0``, …)
            * ``Dict[str, str]`` — ``{id: sequence}`` mapping
            * ``List[SeqRecord]`` — BioPython records; ``.id`` and ``.seq`` are used
        simplify_io:
            When ``True``, topology strings use generic ``I``/``O`` labels
            instead of membrane-specific characters.

        Returns
        -------
        PredictionCollection
        """
        # --- normalise input to list of (id, sequence_str) tuples -----------
        if isinstance(sequences, dict):
            id_seq_pairs = [(k, str(v)) for k, v in sequences.items()]
        else:
            id_seq_pairs = []
            for i, s in enumerate(sequences):
                if isinstance(s, str):
                    id_seq_pairs.append((f"seq_{i}", s))
                else:  # assume SeqRecord or any object with .id / .seq
                    id_seq_pairs.append((str(s.id), str(s.seq)))
        # ---------------------------------------------------------------------
        predictions = []

        with tqdm(total=len(id_seq_pairs), desc="Predicting", unit="seq", disable=not progress_bar) as pbar:
            for seq_id, sequence in id_seq_pairs:
                # --- embedding ---
                embedding = get_embedding(
                    sequence, self._esm_model, self._esm_alphabet, self._device
                ).unsqueeze(0)
                mask = torch.ones(1, len(sequence), device=self._device)

                # --- ensemble inference ---
                all_topo, all_mem_probs, all_marginals = [], [], []
                for i in range(len(self._models.topology_models)):
                    t, m_p, marg, _ = self._models.predict_single_model(i, embedding, mask, compute_marginals=compute_marginals)
                    all_topo.append(t)
                    all_mem_probs.append(m_p)
                    all_marginals.append(marg)

                # --- aggregation ---
                final_topo_ab, final_marginals = aggregate_topology_predictions(
                    all_topo, all_marginals
                )
                pred_mem_indices, mem_probs = aggregate_memtype_predictions(all_mem_probs)
                final_topo_ab = suppress_implausible_predictions(final_topo_ab)

                # --- postprocessing ---
                type_code = infer_structural_type(final_topo_ab)
                is_non_tm = type_code in ("I", "S", "T")
                main_mem_idx = max(pred_mem_indices, key=lambda i: mem_probs[i])

                # Non-TM proteins get no membrane type assignment
                if is_non_tm:
                    predicted_mem_names = []
                    mem_probs = np.zeros_like(mem_probs)
                    effective_simplify = True   # always use I/O for non-TM
                else:
                    predicted_mem_names = [MEMBRANE_TYPE_NAMES[i] for i in pred_mem_indices]
                    effective_simplify = simplify_io

                final_topo_out = translate_ab_topology(
                    final_topo_ab, main_mem_idx, simplify_io=effective_simplify
                )
                named_regions = get_named_regions(
                    final_topo_ab, main_mem_idx, simplify_io=effective_simplify
                )

                predictions.append(Prediction(
                    id=seq_id,
                    sequence=sequence,
                    type=DISPLAY_PROT_TYPE_MAPPING[type_code],
                    membrane_types=predicted_mem_names,
                    membrane_types_probabilities=mem_probs,
                    topology_string=final_topo_out,
                    segments=named_regions,
                    _topology_ab=final_topo_ab,
                    _marginals=final_marginals,
                    _main_mem_idx=main_mem_idx,
                    _simplify_io=effective_simplify,
                ))

                pbar.update(1)

        return PredictionCollection(predictions)
