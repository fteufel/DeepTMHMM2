"""
Constants and static mappings for the DeepTMHMM 2 predictor.
Contains membrane type names, CRF configurations, and label translations.

Model checkpoint paths are managed by weights.py.
"""

from typing import List, Dict, Any

# Supported membrane type names and their indices in the model
MEMBRANE_TYPE_NAMES: Dict[int, str] = {
    0: "Archaebacterial membrane",
    1: "Bacterial Gram-negative inner membrane",
    2: "Bacterial Gram-negative outer membrane",
    3: "Eukaryotic plasma membrane",
    4: "Mitochondrial inner membrane",
    5: "Endoplasmic reticulum membrane",
    6: "Thylakoid membrane",
    7: "Bacterial Gram-positive plasma membrane",
    8: "Golgi membrane",
    9: "Nuclear inner membrane",
    # 10: Peroxisome membrane (Not supported)
    11: "Endosome membrane",
    12: "Vacuole membrane",
    13: "Vesicle membrane",
    14: "Viral membrane",
    # 15: Chloroplast inner membrane (Not supported)
    16: "Chloroplast outer membrane",
    17: "Mitochondrial outer membrane",
    18: "Lysosome membrane",
    # 19: Nuclear outer membrane (Not supported)
    # 20: Cytoplasmic granule membrane (Not supported)
}

# Empirical thresholds for supported membrane types (multi-label)
MEMBRANE_TYPE_THRESHOLDS: Dict[int, float] = {
    0: 0.717,  # Archaebacterial membrane
    1: 0.374,  # Bacterial Gram-negative inner membrane
    2: 0.899,  # Bacterial Gram-negative outer membrane
    3: 0.172,  # Eukaryotic plasma membrane
    4: 0.323,  # Mitochondrial inner membrane
    5: 0.182,  # Endoplasmic reticulum membrane
    6: 0.535,  # Thylakoid membrane
    7: 0.485,  # Bacterial Gram-positive plasma membrane
    8: 0.818,  # Golgi membrane
    9: 0.040,  # Nuclear inner membrane
    11: 0.020, # Endosome membrane
    12: 0.101, # Vacuole membrane
    13: 0.303, # Vesicle membrane
    14: 0.535, # Viral membrane
    16: 0.606, # Chloroplast outer membrane
    17: 0.303, # Mitochondrial outer membrane
    18: 0.010, # Lysosome membrane
}

# Mapping from raw marginal probability indices to plotting categories
# 0: I/O (merged), 2: Signal, 3: Alpha, 4: Beta, 5: Reentrant, 6: Interfacial, 7: Transit
MARGINAL_CATEGORY_MAPPING: Dict[int, int] = {
    0: 0, 1: 0, # Sides
    2: 2,       # Signal
    3: 3,       # Alpha
    4: 4,       # Beta
    5: 5,       # Reentrant
    6: 6,       # Interfacial
    7: 7        # Transit
}

CRF_CONFIG: Dict[str, Any] = {
    'type_label_to_int': {
        'I': 0, 'S': 1, 'M': 2, 'M+S': 3, 'B': 4, 'M+R': 5, 'M+F': 6, 'M+R+F': 7, 'M+T': 8, 'T': 9
    },
    'maximum_segment_lengths': {
        'S': -1, '>': -1, 'M': 30, 'B': 15, 'R': -1, 'F': -1, 'Mab': 30, 'Mba': 30,
        'Bab': 15, 'Bba': 15, 'Raa': -1, 'Rbb': -1, 'Faa': -1, 'Fbb': -1
    },
    'minimum_segment_lengths': {
        'S': 10, '>': 5, 'M': 10, 'B': 5, 'R': 8, 'F': 5, 'Mab': 10, 'Mba': 10,
        'Bab': 5, 'Bba': 5, 'Raa': 8, 'Rbb': 8, 'Faa': 5, 'Fbb': 5
    },
    'use_start_end_transitions': True
}

MEMBRANE_TOPOLOGY_MAP: Dict[int, Dict[str, str]] = {
    0: {'a': 'A', 'b': 'a'}, # Archaebacterial membrane|cytoplasmic/extracellular
    1: {'a': 'C', 'b': 'c'}, # Bacterial Gram-negative inner membrane|cytoplasmic/periplasm
    2: {'a': 'D', 'b': 'd'}, # Bacterial Gram-negative outer membrane|periplasm/extracellular
    3: {'a': 'E', 'b': 'e'}, # Eukaryotic plasma membrane cytoplasmic side extracellular side
    4: {'a': 'G', 'b': 'g'}, # Mitochondrial inner membrane mitochondrial matrix intermembrane space
    5: {'a': 'H', 'b': 'h'}, # Endoplasmic reticulum membrane cytoplasmic side lumenal side
    6: {'a': 'J', 'b': 'j'}, # Thylakoid membrane stroma thylakoid space
    7: {'a': 'K', 'b': 'k'}, # Bacterial Gram-positive plasma membrane|cytoplasmic/periplasm
    8: {'a': 'L', 'b': 'l'}, # Golgi membrane cytoplasmic side lumenal side
    9: {'a': 'N', 'b': 'n'}, # Nuclear inner membrane nucleus matrix side intermembrane space
    10: {'a': 'Z', 'b': 'z'}, # Peroxisome membrane cytoplasmic side lumenal side
    11: {'a': 'P', 'b': 'p'}, # Endosome membrane cytoplasmic side lumenal side
    12: {'a': 'Q', 'b': 'q'}, # Vacuole membrane cytoplasmic side lumenal side
    13: {'a': 'T', 'b': 't'}, # Vesicle membrane cytoplasmic side lumenal side
    14: {'a': 'V', 'b': 'v'}, # Viral membrane intravirion side virion surface
    15: {'a': 'W', 'b': 'w'}, # Chloroplast inner membrane stroma intermembrane space
    16: {'a': 'X', 'b': 'x'}, # Chloroplast outer membrane cytoplasmic side intermembrane space
    17: {'a': 'Y', 'b': 'y'}, # Mitochondrial outer membrane cytoplasmic side intermembrane space
    18: {'a': '3', 'b': '4'}, # Lysosome membrane cytoplasmic side lumenal side
    19: {'a': '5', 'b': '6'}, # Nuclear outer membrane cytoplasmic side intermembrane space
    20: {'a': '7', 'b': '8'}, # Cytoplasmic granule membrane cytoplasmic side intermembrane space
}

MEMBRANE_REGION_NAMES: Dict[int, Dict[str, str]] = {
    0: {'a': 'Cytoplasmic', 'b': 'Extracellular'},
    1: {'a': 'Cytoplasmic', 'b': 'Periplasmic'},
    2: {'a': 'Periplasmic', 'b': 'Extracellular'},
    3: {'a': 'Cytoplasmic', 'b': 'Extracellular'},
    4: {'a': 'Mitochondrial matrix', 'b': 'Intermembrane space'},
    5: {'a': 'Cytoplasmic', 'b': 'Lumenal'},
    6: {'a': 'Stroma', 'b': 'Thylakoid space'},
    7: {'a': 'Cytoplasmic', 'b': 'Periplasmic'},
    8: {'a': 'Cytoplasmic', 'b': 'Lumenal'},
    9: {'a': 'Nucleus matrix', 'b': 'Intermembrane space'},
    10: {'a': 'Cytoplasmic', 'b': 'Lumenal'},
    11: {'a': 'Cytoplasmic', 'b': 'Lumenal'},
    12: {'a': 'Cytoplasmic', 'b': 'Lumenal'},
    13: {'a': 'Cytoplasmic', 'b': 'Lumenal'},
    14: {'a': 'Intravirion', 'b': 'Virion surface'},
    15: {'a': 'Stroma', 'b': 'Intermembrane space'},
    16: {'a': 'Cytoplasmic', 'b': 'Intermembrane space'},
    17: {'a': 'Cytoplasmic', 'b': 'Intermembrane space'},
    18: {'a': 'Cytoplasmic', 'b': 'Lumenal'},
    19: {'a': 'Cytoplasmic', 'b': 'Intermembrane space'},
    20: {'a': 'Cytoplasmic', 'b': 'Intermembrane space'},
}

# User-facing structural type mapping
DISPLAY_PROT_TYPE_MAPPING = {
    'M': 'Alpha TM',
    'M+S': 'Alpha TM + SP',
    'S': 'Globular + SP',
    'I': 'Globular',
    'B': 'Beta Barrel',
    'M+R': 'Alpha TM',
    'M+F': 'Alpha TM',
    'M+R+F': 'Alpha TM',
    'M+T': 'Alpha TM + TP',
    'T': 'Globular + TP'
}
