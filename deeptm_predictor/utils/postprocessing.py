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


def convert_to_segments(label_in: str) -> List[Tuple[str, int, int]]:
    """
    Convert a label sequence (string or numpy array) to a list of (label, start, end) segments.
    """
    segments = []
    iterable = enumerate(label_in)

    for label, group in groupby(iterable, key=lambda x: x[1]):
        group_list = list(group)
        start_idx = group_list[0][0]
        end_idx = group_list[-1][0] + 1
        segments.append((label, start_idx, end_idx))
    return segments


def is_topologies_equal(
    topology_a: List[Tuple[str,int,int]], 
    topology_b: List[Tuple[str,int,int]], 
    minimum_seqment_overlap=5
):
    labels_to_check = ('M', 'B', "R", "F") # S, T are checked without overlap by the simple label equality check

    if len(topology_a) != len(topology_b):
        return False
    for idx, (label_a_char, label_a_start, label_a_end) in enumerate(topology_a):
        label_b_char, label_b_start, label_b_end = topology_b[idx] # Unpack current segment from topology_b

        if label_a_char != label_b_char:
            # mismatch - the two topologies have a different segment here.
            return False
        
        # Only perform overlap check for specific labels ('M', 'B', 'R', 'F')
        if label_a_char in labels_to_check:
            # Ensure we don't go out of bounds for the next segment if it's the last one
            if idx + 1 >= len(topology_a) or idx + 1 >= len(topology_b):
                # If it's the last segment of a type that requires overlap check,
                # and the labels matched, we consider this a match for the segment.
                continue

            # Calculate overlap of current segments based on start and end positions
            overlap_segment_start = max(label_a_start, label_b_start)
            overlap_segment_end = min(label_a_end, label_b_end)
            
            current_minimum_segment_overlap = minimum_seqment_overlap
            if label_a_char == 'B': # Special case for Beta regions
                current_minimum_segment_overlap = 3
            
            if overlap_segment_end - overlap_segment_start < current_minimum_segment_overlap:
                return False
    return True



def aggregate_topology_predictions(
    all_topology_preds: List[str], 
    all_marginal_probs: List[Dict[int, np.ndarray]]
) -> Tuple[str, Dict[int, np.ndarray]]:
    """
    Ensemble aggregation for topology and marginal probabilities.
    Maps raw state marginals to simplified categories for plotting.
    """

    all_topologies_structured = [convert_to_segments(topology) for topology in all_topology_preds]
    
    agreements = [0] * len(all_topologies_structured)
    for i in range(len(all_topologies_structured)):
        for j in range(len(all_topologies_structured)):
            if i != j and is_topologies_equal(all_topologies_structured[i], all_topologies_structured[j]):
                agreements[i] += 1
    
    max_agreements = max(agreements)
    best_indices = [i for i, count in enumerate(agreements) if count == max_agreements]    
    # Tie-breaking: picks the first among the tied models
    best_idx = best_indices[0]
    final_topology = all_topology_preds[best_idx]
    final_marginals = all_marginal_probs[best_idx]
            
    return final_topology, final_marginals

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
    

def suppress_implausible_predictions(topology: str) -> str:
    """Suppress implausible predictions in the topology string."""
    
    if topology.count('S') > 70:
        topology = topology.replace('S', 'b')
    if topology.count('>') > 170:
        topology = topology.replace('>', 'b')
    return topology