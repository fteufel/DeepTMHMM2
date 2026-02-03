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

def _plot_conditional_fill(ax, lab_y, x_range, target_label, fill_color, 
                           fill_y_inside=(0, 1), fill_y_outside=(1, 2)):
    """
    Helper function to plot regions with conditional fill based on preceding residue.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    lab_y : np.ndarray
        Array of numerical labels for topology.
    x_range : np.ndarray
        Array of x-coordinates (amino acid positions).
    target_label : int
        The numerical label for the region type to be conditionally filled (e.g., 6 for interfacial, 7 for re-entrant).
    fill_color : str
        Color for the fill.
    fill_y_inside : tuple
        (y_bottom, y_top) for fill when preceded by 'inside' (lab_y == 0).
    fill_y_outside : tuple
        (y_bottom, y_top) for fill when preceded by 'outside' (lab_y == 1).
    """
    target_indices = np.where(lab_y == target_label)[0]

    for k, g in groupby(enumerate(target_indices), lambda ix: ix[0] - ix[1]):
        segment_indices = list(map(lambda ix: ix[1], g))
        
        if not segment_indices:
            continue

        start_idx = segment_indices[0]
        end_idx = segment_indices[-1]
        
        preceding_state = None
        if start_idx > 0:
            preceding_state = lab_y[start_idx - 1]
        
        y_bottom, y_top = fill_y_inside # Default for ambiguous or start-of-sequence cases

        if preceding_state == 0: # Preceded by 'inside'
            y_bottom, y_top = fill_y_inside
        elif preceding_state == 1: # Preceded by 'outside'
            y_bottom, y_top = fill_y_outside
        
        segment_x_range = x_range[start_idx : end_idx + 1]
        ax.fill_between(segment_x_range, y_bottom, y_top, 
                         color=fill_color, alpha=0.8)


def plot_prediction(seq_id: str, 
                    sequence: str, 
                    topology: str, 
                    mem_type_name: str, 
                    agg_probs: Optional[Dict[int, np.ndarray]], 
                    output_dir: Path):
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
    """
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
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

    # Create y-values for plotting each compartment/region
    plot_inside = np.where(lab_y == 0, 0, np.nan)
    plot_outside = np.where(lab_y == 1, 2, np.nan)
    plot_signal = np.where(lab_y == 3, 1, np.nan)

    # Plot lines for soluble regions
    ax0.plot(x_range, plot_outside, COLORS['outside'], linewidth=3)
    ax0.plot(x_range, plot_inside, COLORS['inside'], linewidth=3)
    ax0.plot(x_range, plot_signal, COLORS['signal'], linewidth=3)

    # Fill for transmembrane regions
    is_membrane = (lab_y == 4)
    is_beta = (lab_y == 5)
    ax0.fill_between(x_range, 0, 2, where=is_membrane, color=COLORS['alpha'], alpha=0.8)
    ax0.fill_between(x_range, 0, 2, where=is_beta, color=COLORS['beta'], alpha=0.8)

    # Call helper function for re-entrant regions
    _plot_conditional_fill(ax0, lab_y, x_range, 7, COLORS['reentrant'], 
                           fill_y_inside=(0, 1), fill_y_outside=(1, 2))

    # Call helper function for interfacial regions
    _plot_conditional_fill(ax0, lab_y, x_range, 6, COLORS['interfacial'], 
                           fill_y_inside=(0, 0.5), fill_y_outside=(1.5, 2))

    # Set y-ticks and labels based on inferred protein type
    ax0.set_ylim(-0.5, 2.5)

    is_transit = topology.startswith('>')
    sp_label = "Transit" if is_transit else "Signal"

    if "BETA" in prot_type_str:
        ax0.set_yticks([0, 1, 2], ['Inside', 'Beta Barrel', 'Outside'])
    elif "SP" in prot_type_str and "TM" in prot_type_str:
        ax0.set_yticks([0, 1, 2], ['Inside', f'{sp_label}/TM', 'Outside'])
    elif "SP" in prot_type_str:
        ax0.set_yticks([0, 1, 2], ['Inside', sp_label, 'Outside'])
    elif "TM" in prot_type_str:
         ax0.set_yticks([0, 1, 2], ['Inside', 'TM', 'Outside'])
    else:
        ax0.set_yticks([0, 2], ['Inside', 'Outside'])

    ax0.set_xlim(0, len(sequence))
    ax0.set_title(f"DeepTM: {seq_id} | Type: {prot_type_str} | {mem_type_name}", fontsize=14, fontweight='bold')
    
    # Track 2: Posterior Probabilities (Continuous lines)
    ax1 = plt.subplot(gs[1])
    
    if agg_probs:
        cat_meta = {
            0: {'label': 'Inside/Outside', 'color': COLORS['inside']},
            2: {'label': 'Signal Peptide', 'color': COLORS['signal']},
            3: {'label': 'Beta Barrel', 'color': COLORS['beta']},
            4: {'label': 'Alpha TM', 'color': COLORS['alpha']},
            5: {'label': 'Reentrant', 'color': COLORS['reentrant']},
            6: {'label': 'Interfacial', 'color': COLORS['interfacial']}, 
            7: {'label': 'Transit Peptide', 'color': COLORS['transit']}
        }
        
        for cat_idx, probs in agg_probs.items():
            if cat_idx in cat_meta:
                meta = cat_meta[cat_idx]
                ax1.plot(x_range, probs, label=meta['label'], color=meta['color'], linewidth=2)
        
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_ylabel("Posterior Probability", fontsize=12)
        ax1.legend(loc='upper right', frameon=True, shadow=True)
    else:
        ax1.text(0.5, 0.5, "Probability data not provided", ha='center', va='center')

    ax1.set_xlim(0, len(sequence))
    ax1.set_xlabel("Amino Acid Position", fontsize=12)
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{seq_id}_plot.png", dpi=200)
    plt.close()
