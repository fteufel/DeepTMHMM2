"""
Visualization utilities for DeepTM predictions.
Generates horizontal bar plots for topology predictions with posterior probability tracks.
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from itertools import groupby
from pathlib import Path
from typing import Dict, Optional

from ..constants import MEMBRANE_REGION_NAMES

COLORS = {
    'inside': '#377eb8',
    'outside': '#f781bf',
    'signal': '#ff7f00',
    'alpha': '#e41a1c',
    'beta': '#f28e2b',
    'reentrant': '#4daf4a',
    'interfacial': '#984ea3',
    'transit': '#e0e010'
}




def plot_prediction(seq_id: str, 
                    sequence: str, 
                    topology: str, 
                    mem_type_name: str, 
                    agg_probs: Optional[Dict[int, np.ndarray]], 
                    output_dir: Path,
                    main_mem_idx: Optional[int] = None,
                    simplify_io: bool = False):
    """
    Generates a visual plot of the predicted topology and posterior probabilities.
    
    Parameters
    ----------
    seq_id : str
        Protein identifier.
    sequence : str
        Amino acid sequence.
    topology : str
        Predicted topology string (e.g. 'aMBS..')
    mem_type_name : str
        Header string for the plot title (e.g. joined membrane type names).
    agg_probs : Dict[int, np.ndarray], optional
        Aggregated marginal probabilities for categories (0: I/O, 2: Signal, 3: Alpha, etc.).
    output_dir : Path
        Directory to save the plot.
    main_mem_idx : int, optional
        Index into MEMBRANE_REGION_NAMES for the primary predicted membrane type.
        Used to label the y-axis with biologically accurate compartment names.
    simplify_io : bool
        If True, use generic 'Inside'/'Outside' labels regardless of membrane type.
    """
    
    if agg_probs:
        plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
    else:
        plt.figure(figsize=(12, 1.6))
        gs = gridspec.GridSpec(1, 1)

    x_range = np.arange(len(sequence))

    # Track 1: Predicted Topology (Compartment plot)
    ax0 = plt.subplot(gs[0])

    # Convert topology string to numerical labels based on reference logic
    # a=inside(0), b=outside(1), S=signal(3), M=membrane(4), B=beta(5), R=reentrant(6), F=interfacial(6), >=transit(3)
    label_map = {'a': 0, 'b': 1, 'S': 3, 'M': 4, 'B': 5, 'R': 7, 'F': 6, '>': 3} 
    lab_y = np.array([label_map.get(char, -1) for char in topology])

    # Infer protein type for plot labels
    prot_types = []
    if 3 in lab_y: prot_types.append("SP")
    if 5 in lab_y: prot_types.append("BETA")
    if 4 in lab_y: prot_types.append("TM")
    prot_type_str = "+".join(prot_types) if prot_types else "Globular"

    max_len = len(sequence)

    for k, g in groupby(enumerate(lab_y), lambda x: x[1]):
        segment = list(g)
        if not segment:
            continue
        start_idx = segment[0][0]
        end_idx = segment[-1][0]
        label = k
        
        x_start = start_idx - 0.5
        x_end = end_idx + 0.5
        x_seg = [x_start, x_end]
        
        if label == 0:
            ax0.plot(x_seg, [0, 0], color=COLORS['inside'], linewidth=3, solid_capstyle='butt')
        elif label == 1:
            ax0.plot(x_seg, [2, 2], color=COLORS['outside'], linewidth=3, solid_capstyle='butt')
        elif label == 3:
            ax0.plot(x_seg, [1, 1], color=COLORS['signal'], linewidth=3, solid_capstyle='butt')
        elif label == 4:
            ax0.fill_between(x_seg, [0, 0], [2, 2], color=COLORS['alpha'], alpha=0.8, linewidth=0)
        elif label == 5:
            ax0.fill_between(x_seg, [0, 0], [2, 2], color=COLORS['beta'], alpha=0.8, linewidth=0)
        elif label == 7:
            preceding_state = lab_y[start_idx - 1] if start_idx > 0 else 0
            y_bottom, y_top = (1, 2) if preceding_state == 1 else (0, 1)
            ax0.fill_between(x_seg, [y_bottom, y_bottom], [y_top, y_top], color=COLORS['reentrant'], alpha=0.8, linewidth=0)
        elif label == 6:
            preceding_state = lab_y[start_idx - 1] if start_idx > 0 else 0
            y_bottom, y_top = (1.5, 2) if preceding_state == 1 else (0, 0.5)
            ax0.fill_between(x_seg, [y_bottom, y_bottom], [y_top, y_top], color=COLORS['interfacial'], alpha=0.8, linewidth=0)

    # Set y-ticks and labels based on inferred protein type
    ax0.set_ylim(-0.5, 2.5)

    is_transit = topology.startswith('>')
    sp_label = "TP" if is_transit else "SP"

    # Resolve side labels: use biological compartment names when available and not simplified
    is_tm_type = "TM" in prot_type_str or "BETA" in prot_type_str
    if (not simplify_io
            and is_tm_type
            and main_mem_idx is not None
            and main_mem_idx in MEMBRANE_REGION_NAMES):
        region = MEMBRANE_REGION_NAMES[main_mem_idx]
        side_a = region['a']   # topological 'inside' (a-state)
        side_b = region['b']   # topological 'outside' (b-state)
    else:
        side_a, side_b = 'Inside', 'Outside'

    if "BETA" in prot_type_str:
        ax0.set_yticks([0, 1, 2], [side_a, f'{sp_label}/Membrane', side_b])
    elif "SP" in prot_type_str and "TM" in prot_type_str:
        ax0.set_yticks([0, 1, 2], [side_a, f'{sp_label}/Membrane', side_b])
    elif "SP" in prot_type_str:
        ax0.set_yticks([0, 1, 2], [side_a, sp_label, side_b])
    elif "TM" in prot_type_str:
        ax0.set_yticks([0, 1, 2], [side_a, 'Membrane', side_b])
    else:
        ax0.set_yticks([0, 2], ['Inside', 'Outside'])

    ax0.set_xlim(0, len(sequence))
    if mem_type_name == "":
        ax0.set_title(f"{seq_id} | Type: {prot_type_str}", fontsize=14, fontweight='bold')
    else:
        ax0.set_title(f"{seq_id} | Type: {prot_type_str} | {mem_type_name}", fontsize=14, fontweight='bold')
    
    if agg_probs:
        # Track 2: Posterior Probabilities (Continuous lines)
        ax1 = plt.subplot(gs[1])
        cat_meta = {
            0: {'label': 'Inside/Outside', 'color': COLORS['inside']},
            2: {'label': 'Signal Peptide', 'color': COLORS['signal']},
            3: {'label': 'Beta Barrel', 'color': COLORS['beta']},
            4: {'label': 'Alpha TM', 'color': COLORS['alpha']},
        }

        alpha_only = {
            5: {'label': 'Reentrant', 'color': COLORS['reentrant']},
            6: {'label': 'Interfacial', 'color': COLORS['interfacial']}, 
            7: {'label': 'Transit Peptide', 'color': COLORS['transit']}
        }
        if not "BETA" in prot_type_str:
            cat_meta.update(alpha_only)
        
        for cat_idx, probs in agg_probs.items():
            if cat_idx in cat_meta:
                meta = cat_meta[cat_idx]
                ax1.plot(x_range, probs, label=meta['label'], color=meta['color'], linewidth=2)
        
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_ylabel("Posterior Probability", fontsize=12)
        ax1.legend(loc='upper right', frameon=True, shadow=False)

        ax1.set_xlim(0, len(sequence))
        ax1.set_xlabel("Amino Acid Position", fontsize=12)
        ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{seq_id}_plot.png", dpi=200)
    plt.close()
