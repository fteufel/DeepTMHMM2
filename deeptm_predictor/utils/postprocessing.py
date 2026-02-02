"""
Post-processing utilities for DeepTM predictions.
Includes topology voting, membrane type thresholding, and label translation.
"""

import torch
import numpy as np
from itertools import groupby
from typing import List, Tuple, Dict
from ..constants import (
    MEMBRANE_TYPE_THRESHOLDS, 
    MEMBRANE_TOPOLOGY_MAP, 
    MEMBRANE_REGION_NAMES,
    MARGINAL_CATEGORY_MAPPING
)

def infer_structural_type(topology: str) -> str:
    """Infer a simple structural type from the predicted topology string."""
    if 'B' in topology:
        return 'B'
    if 'M' in topology:
        if 'S' in topology:
            return 'M+S'
        if '>' in topology:
            return 'M+T'
        return 'M'
    if 'S' in topology:
        return 'S'
    if '>' in topology:
        return 'T'
    return 'I'

def translate_ab_topology(topology: str, mem_type_idx: int, simplify_io: bool = False) -> str:
    """
    Translates generic a/b labels to either:
    - membrane-specific characters (if simplify_io is False)
    - generic i/o labels (if simplify_io is True)
    """
    if simplify_io:
        translation_map = {'a': 'i', 'b': 'o'}
    elif mem_type_idx in MEMBRANE_TOPOLOGY_MAP:
        translation_map = MEMBRANE_TOPOLOGY_MAP[mem_type_idx]
    else:
        return topology
        
    return "".join([translation_map.get(c, c) for c in topology])

def get_region_name(label: str, mem_type_idx: int, simplify_io: bool = False) -> str:
    """Maps residue-level labels to descriptive region names."""
    name_map = {'S': 'signal', 'M': 'TMhelix', 'B': 'Beta sheet', 'R': 'reentrant', 'F': 'interfacial', '>': 'transit_peptide'}
    if label in name_map:
        return name_map[label]
    
    if simplify_io:
        return "inside" if label == 'a' else "outside"
    
    if mem_type_idx in MEMBRANE_REGION_NAMES:
        return MEMBRANE_REGION_NAMES[mem_type_idx].get(label, label)
    
    return label

def get_named_regions(topology_ab: str, mem_type_idx: int, simplify_io: bool = False) -> List[Tuple[str, int, int]]:
    """
    Groups the topology string into segments and resolves their biological names.
    This is the final 'renaming' step before delivery to output formatters.
    """
    segments = []
    pos = 1
    for label, group in groupby(topology_ab):
        length = len(list(group))
        name = get_region_name(label, mem_type_idx, simplify_io)
        segments.append((name, pos, pos + length - 1))
        pos += length
    return segments

def aggregate_topology_predictions(
    all_topology_preds: List[str], 
    all_marginal_probs: List[torch.Tensor]
) -> Tuple[str, Dict[int, np.ndarray]]:
    """
    Ensemble aggregation for topology and marginal probabilities.
    Maps raw state marginals to simplified categories for plotting.
    """
    agreements = [0] * len(all_topology_preds)
    for i in range(len(all_topology_preds)):
        for j in range(len(all_topology_preds)):
            if i != j and all_topology_preds[i] == all_topology_preds[j]:
                agreements[i] += 1
    
    max_agreements = max(agreements)
    best_indices = [i for i, count in enumerate(agreements) if count == max_agreements]
    
    # Tie-breaking: picks the first among the tied models
    best_idx = best_indices[0]
    final_topology = all_topology_preds[best_idx]
    final_marginals_raw = all_marginal_probs[best_idx]
    
    # Aggregate marginals using the static categorical mapping
    category_probs = {}
    for state_idx, cat_idx in MARGINAL_CATEGORY_MAPPING.items():
        probs = final_marginals_raw[:, state_idx].cpu().numpy()
        if cat_idx not in category_probs:
            category_probs[cat_idx] = probs
        else:
            category_probs[cat_idx] += probs
            
    return final_topology, category_probs

def aggregate_memtype_predictions(all_memtype_probs: List[torch.Tensor]) -> Tuple[List[int], torch.Tensor]:
    """Averaging and thresholding for multi-label membrane type predictions."""
    avg_probs = torch.stack(all_memtype_probs).mean(dim=0)
    probs_np = avg_probs.cpu().numpy().squeeze()
    
    all_predicted_indices = []
    supported_probs = {}

    
    for idx in MEMBRANE_TYPE_THRESHOLDS:
        supported_probs[idx] = probs_np[idx]
        if probs_np[idx] >= MEMBRANE_TYPE_THRESHOLDS[idx]:
            all_predicted_indices.append(idx)
            
    # If no type passes its threshold, return the argmax of supported types
    if not all_predicted_indices:
        if supported_probs:
            main_idx = max(supported_probs, key=supported_probs.get)
            all_predicted_indices = [main_idx]
            
    return all_predicted_indices, probs_np