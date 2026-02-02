
'''
The configuration of the CRF state space.

This state space implements the following:

- Signal S (min_len to max_len)
- Transit peptide ">: (min_len to max_len). Only for alpha proteins.
- Four "sides" a_alpha + b_alpha + a_beta + b_beta
- Beta proteins B (beta a>b and beta b>a), with min_len and max_len
- Alpha proteins M (alpha a>b and alpha b>a), with min_len and max_len
- Reentrant regions R (reentrant a>a and reentrant b>b), with min_len and max_len. Only for alpha proteins.
- interfactial helices I (interfactial a>a and interfacial b>b), with min_len and max_len. Only for alpha proteins.

- Valid starting points are S, a, b
- Valid end points are a, b

- Side "b" is the side that can have a signal peptide. (typically "O" in default TM prediction).
- S is always optional regardless of the type (wrong biology, but enables correct predictions on wrongly annotated N termini)
- > can go to side a or b. TPs are not predictive of Nterm orientation.   
- > cannot go to beta barrel - 0 observed cases in the data.  
'''

import math
import numpy as np
import torch
from typing import Dict, Union, List, Any, Tuple
from itertools import groupby



# Segment lengths
# max_signal_length = 60
# max_alpha_membrane_length = 30
# max_beta_membrane_length = 15
# max_
MINIMUM_SEGMENT_LENGTHS = {
    'S': 10,   # Signal
    '>': 10, # Transit peptide
    'M': 10,
    'B': 5,
    'R': 8,
    'F': 6,
    'Mab': 10, # Alpha TM
    'Mba': 10, # Alpha TM
    'Bab': 5,  # Beta TM
    'Bba': 5,  # Beta TM
    'Raa': 8, # Reentrant
    'Rbb': 8, # Reentrant
    'Faa': 6, # Interfacial
    'Fbb': 6, # Interfacial
}

MAXIMUM_SEGMENT_LENGTHS = {
    'S': 60,   # Signal
    '>': 170, # Transit peptide
    'M': 30,
    'B': 15,
    'R': 30,
    'F': -1,
    'Mab': 30, # Alpha TM
    'Mba': 30, # Alpha TM
    'Bab': 15,  # Beta TM
    'Bba': 15,  # Beta TM
    'Raa': 30, # Reentrant
    'Rbb': 30, # Reentrant
    'Faa': -1, # Interfacial
    'Fbb': -1, # Interfacial
}




class SplitIOCRFwithTPConfig:
    def __init__(
            self,
            maximum_segment_lengths: Dict[str, int] = None,
            minimum_segment_lengths: Dict[str, int] = None,
            use_start_end_transitions: bool = True,
            type_label_to_int: Dict[str, int] = None,
            undetermined_mode: str = 'segments'
            ):
        """
        A 4-side CRF configuration with signal, transit, alpha, beta, reentrant and interfacial regions.
        
        Parameters
        ----------
        maximum_segment_lengths : Dict[str, int], optional
            The maximum length of each segment type, by default None
        minimum_segment_lengths : Dict[str, int], optional
            The minimum length of each segment type, by default None
        use_start_end_transitions : bool, optional
            Whether to constrain the start and end states to the allowed start and end states, by default True
            If this is set to False, we also allow truncated segments in `U` regions at the N and C termini of 
            training samples.
        type_label_to_int : Dict[str, int], optional
            A dictionary that maps the sequence types to integers, by default None
        undetermined_mode : str, optional
            The mode to use for undetermined regions, by default 'segments'
            If 'segments', we add "segments that fit" to the undetermined regions.
            If 'any', we simply allow any state in the undetermined regions (the CRF transition constraints will still apply).
        """

        self.label_to_int = {"a":0 , "b": 1, "a_alpha":0, "b_alpha":1, "a_beta":0, "b_beta":1, "S":2, "M":3, "B":4, "R": 5, "F": 6, '>': 7, 'U': -1, 'u': -1, '.': -1}
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

        self.num_emissions = 10 # -1 because we don't count the "u" label

        self.undetermined_mode = 'segments' # 'segments' or 'any'

        if type_label_to_int is None:
            type_label_to_int = {"I":0, "S":1, "M":2, "M+S":3, "B":4, "M+R":5, "M+F": 6, 'M+R+F': 7, 'M+T': 8, "T": 9}

        self.type_label_to_int = type_label_to_int
        self.type_int_to_label = {v: k for k, v in self.type_label_to_int.items()}
        # check if any values are duplicated and warn that they will be collapsed.
        if len(set(self.type_label_to_int.values())) != len(self.type_label_to_int):
            import warnings
            warnings.warn(f"Duplicate values in type_label_to_int configuration. This will cause the affected types to be collapsed into one type. Make sure this is intentional.")
            # make type_int_to_label again - keep first occurence of each value (dict is insertion ordered by default)
            self.type_int_to_label = {}
            for k, v in self.type_label_to_int.items():
                if v not in self.type_int_to_label:
                    self.type_int_to_label[v] = k
            

        self.crf_states = {}
        self.allowed_state_transitions = []

        if maximum_segment_lengths is None:
            maximum_segment_lengths = MAXIMUM_SEGMENT_LENGTHS

        if minimum_segment_lengths is None:
            minimum_segment_lengths = MINIMUM_SEGMENT_LENGTHS



        self.max_segment_lengths = maximum_segment_lengths
        self.min_segment_lengths = minimum_segment_lengths

        self.crf_states["a_alpha"] = 0
        self.crf_states["b_alpha"] = 1
        self.crf_states["a_beta"] = 2
        self.crf_states["b_beta"] = 3
        crf_next_state = 4

        # build the internal sequence of states of the M, B, I, R regions
        for tm_type in ["S", ">", "Mab", "Mba", "Bab", "Bba", 'Raa', 'Rbb', 'Faa', 'Fbb']:

            max_segment_length = maximum_segment_lengths[tm_type]
            min_segment_length = minimum_segment_lengths[tm_type]

            # if we have a max_segment_length, we add that many states. if it's -1, we add min_segment_length states.
            for i in range(max(max_segment_length, min_segment_length)):
                self.crf_states[tm_type+str(i)] = crf_next_state
                crf_next_state += 1

        # define allowed transition patterns

        self.allowed_state_transitions.append((self.crf_states["a_alpha"], self.crf_states["a_alpha"])) # infinite loop for globular regions
        self.allowed_state_transitions.append((self.crf_states["b_alpha"], self.crf_states["b_alpha"])) # infinite loop for globular regions
        self.allowed_state_transitions.append((self.crf_states["a_beta"], self.crf_states["a_beta"]))
        self.allowed_state_transitions.append((self.crf_states["b_beta"], self.crf_states["b_beta"]))

        last_signal_state = self.crf_states["S"+str(max(maximum_segment_lengths["S"], minimum_segment_lengths["S"]) - 1)]
        self.allowed_state_transitions.append((last_signal_state, self.crf_states["b_alpha"])) # signal ends in side b.
        self.allowed_state_transitions.append((last_signal_state, self.crf_states["b_beta"]))

        last_transit_state = self.crf_states[">"+str(max(maximum_segment_lengths[">"], minimum_segment_lengths[">"]) - 1)]
        self.allowed_state_transitions.append((last_transit_state, self.crf_states["a_alpha"])) # transit peptide ends in side a.
        self.allowed_state_transitions.append((last_transit_state, self.crf_states["b_alpha"])) # transit peptide ends in side b.

        # transition into the membrane.
        self.allowed_state_transitions.append((self.crf_states["b_beta"], self.crf_states["Bba0"]))
        self.allowed_state_transitions.append((self.crf_states["b_alpha"], self.crf_states["Mba0"]))
        self.allowed_state_transitions.append((self.crf_states["a_beta"], self.crf_states["Bab0"]))
        self.allowed_state_transitions.append((self.crf_states["a_alpha"], self.crf_states["Mab0"]))

        # transitions out of the membrane
        self.allowed_state_transitions.append((self.crf_states["Bba" + str(maximum_segment_lengths["Bba"] - 1)], self.crf_states["a_beta"]))
        self.allowed_state_transitions.append((self.crf_states["Mba" + str(maximum_segment_lengths["Mba"] - 1)], self.crf_states["a_alpha"]))
        self.allowed_state_transitions.append((self.crf_states["Bab" + str(maximum_segment_lengths["Bab"] - 1)], self.crf_states["b_beta"]))
        self.allowed_state_transitions.append((self.crf_states["Mab" + str(maximum_segment_lengths["Mab"] - 1)], self.crf_states["b_alpha"]))

        # transitions into reentrant loop / superficial helix
        self.allowed_state_transitions.append((self.crf_states["a_alpha"], self.crf_states["Raa0"]))
        self.allowed_state_transitions.append((self.crf_states["b_alpha"], self.crf_states["Rbb0"]))
        self.allowed_state_transitions.append((self.crf_states["a_alpha"], self.crf_states["Faa0"]))
        self.allowed_state_transitions.append((self.crf_states["b_alpha"], self.crf_states["Fbb0"]))

        # transitions out of reentrant loop / superficial helix
        # max() ensures that if the maximum is undefined (-1) we put the out transition on the last state.
        self.allowed_state_transitions.append((self.crf_states["Raa" + str(max(maximum_segment_lengths["Raa"], minimum_segment_lengths["Raa"]) - 1)], self.crf_states["a_alpha"]))
        self.allowed_state_transitions.append((self.crf_states["Rbb" + str(max(maximum_segment_lengths["Rbb"], minimum_segment_lengths["Rbb"]) - 1)], self.crf_states["b_alpha"]))
        self.allowed_state_transitions.append((self.crf_states["Faa" + str(max(maximum_segment_lengths["Faa"], minimum_segment_lengths["Faa"]) - 1)], self.crf_states["a_alpha"]))
        self.allowed_state_transitions.append((self.crf_states["Fbb" + str(max(maximum_segment_lengths["Fbb"], minimum_segment_lengths["Fbb"]) - 1)], self.crf_states["b_alpha"]))

        for tm_type in ["S", ">", "Mab", "Mba", "Bab", "Bba", 'Raa', 'Rbb', 'Faa', 'Fbb']:


            max_segment_length = maximum_segment_lengths[tm_type] if maximum_segment_lengths[tm_type] != -1 else minimum_segment_lengths[tm_type]
            min_segment_length = minimum_segment_lengths[tm_type]

            for i in range(max_segment_length - 1): # no transition for last label - we just defined that above.
                self.allowed_state_transitions.append((self.crf_states[tm_type + str(i)], self.crf_states[tm_type + str(i + 1)])) # forward transition in internal region state sequence.

                # skip transitions out of the region that enforce the minimum length.
                start_segment_length = math.ceil(minimum_segment_lengths[tm_type] / 2)
                end_segment_length = math.floor(minimum_segment_lengths[tm_type] / 2)
                if i >= start_segment_length + 1 and i <= max_segment_length - end_segment_length:  # minimum segment size of start_segment_length + end_segment_length
                    self.allowed_state_transitions.append((self.crf_states[tm_type + str(start_segment_length - 1)], self.crf_states[tm_type + str(i)]))



            # no max length - the final state just gets a self-loop.
            if maximum_segment_lengths[tm_type] == -1:
                self.allowed_state_transitions.append((self.crf_states[tm_type + str(min_segment_length - 1)], self.crf_states[tm_type + str(min_segment_length - 1)]))



        self.allowed_start_states = [
            self.crf_states["S0"],
            self.crf_states[">0"],
            self.crf_states["a_alpha"],
            self.crf_states["b_alpha"],
            self.crf_states["a_beta"],
            self.crf_states["b_beta"],
        ]
        self.allowed_end_states = [
            self.crf_states["a_alpha"],
            self.crf_states["b_alpha"],
            self.crf_states["a_beta"],
            self.crf_states["b_beta"],
        ]
        self.use_start_end_transitions = use_start_end_transitions

        
        self._validate()


        # make mappings of NN output to CRF states
        self.signal_state_inds = [v for k,v in self.crf_states.items() if k.startswith("S")]
        self.transit_state_inds = [v for k,v in self.crf_states.items() if k.startswith(">")]
        self.beta_state_inds = [v for k,v in self.crf_states.items() if k.startswith("M")]
        self.helix_state_inds = [v for k,v in self.crf_states.items() if k.startswith("B")]
        self.reentrant_state_inds = [v for k,v in self.crf_states.items() if k.startswith("R")]
        self.interfacial_state_inds = [v for k,v in self.crf_states.items() if k.startswith("F")]


        self.simplification_dict = {
            self.crf_states["a_alpha"]: 0,
            self.crf_states["b_alpha"]: 1,
            self.crf_states["a_beta"]: 0,
            self.crf_states["b_beta"]: 1,
        }
        self.simplification_dict.update({x:2 for x in self.signal_state_inds})
        self.simplification_dict.update({x:3 for x in self.helix_state_inds})
        self.simplification_dict.update({x:4 for x in self.beta_state_inds})
        self.simplification_dict.update({x:5 for x in self.reentrant_state_inds})
        self.simplification_dict.update({x:6 for x in self.interfacial_state_inds})
        self.simplification_dict.update({x:7 for x in self.transit_state_inds})


    @staticmethod
    def _get_unique_consecutive_segments(x: List[Any]) -> Tuple[List[Any], List[int], List[int]]:
        '''
        Get repeated segments of same values together with their start and end.
        NOTE the returned end is meant for indexing as [:end] .
        '''
        values = []
        starts = []
        ends = []

        pos = 0
        for k,g in groupby(x):
            values.append(k)
            starts.append(pos)
            length = sum(1 for _ in g)
            ends.append(pos+length)
            pos += length

        return values, starts, ends



    def convert_crf_states_to_labels(self, labels: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert CRF states back to simple {M,B,R,F,S,>,a,b} labels.
        
        Parameters
        ----------
        labels: np.ndarray or torch.Tensor
            A 1D array of CRF state integers.

        Returns
        -------
        np.ndarray or torch.Tensor
            A 1D array of simple {M,B,R,F,S,>,a,b} labels.
        
        """
        if isinstance(labels, torch.Tensor):
            labels_remapped = torch.ones_like(labels) *-1 #use an impossible value instead of 0/1 so that we catch problems.
        else:
            labels_remapped = np.ones_like(labels) *-1 #use an impossible value instead of 0/1 so that we catch problems.
        
        labels_remapped[labels==self.crf_states['a_alpha']] = self.label_to_int["a_alpha"] 
        labels_remapped[labels==self.crf_states['b_alpha']] = self.label_to_int["b_alpha"]
        labels_remapped[labels==self.crf_states['a_beta']] = self.label_to_int["a_beta"]
        labels_remapped[labels==self.crf_states['b_beta']] = self.label_to_int["b_beta"]


        max_len = self.max_segment_lengths["S"] if self.max_segment_lengths["S"] != -1 else self.min_segment_lengths["S"]
        labels_remapped[(labels >= self.crf_states["S0"]) & (labels <= self.crf_states["S"+str(max_len-1)])] = self.label_to_int["S"]
        max_len = self.max_segment_lengths[">"] if self.max_segment_lengths[">"] != -1 else self.min_segment_lengths[">"]
        labels_remapped[(labels >= self.crf_states[">0"]) & (labels <= self.crf_states[">"+str(max_len-1)])] = self.label_to_int[">"]

        labels_remapped[(labels >= self.crf_states["Mab0"]) & (labels <= self.crf_states["Mab"+str(self.max_segment_lengths["Mab"]-1)])] = self.label_to_int["M"]
        labels_remapped[(labels >= self.crf_states["Mba0"]) & (labels <= self.crf_states["Mba"+str(self.max_segment_lengths["Mba"]-1)])] = self.label_to_int["M"]
        labels_remapped[(labels >= self.crf_states["Bab0"]) & (labels <= self.crf_states["Bab"+str(self.max_segment_lengths["Bab"]-1)])] = self.label_to_int["B"]
        labels_remapped[(labels >= self.crf_states["Bba0"]) & (labels <= self.crf_states["Bba"+str(self.max_segment_lengths["Bba"]-1)])] = self.label_to_int["B"]
        
        max_len = self.max_segment_lengths["Raa"] if self.max_segment_lengths["Raa"] != -1 else self.min_segment_lengths["Raa"]
        labels_remapped[(labels >= self.crf_states["Raa0"]) & (labels <= self.crf_states["Raa"+str(max_len-1)])] = self.label_to_int["R"]
        max_len = self.max_segment_lengths["Rbb"] if self.max_segment_lengths["Rbb"] != -1 else self.min_segment_lengths["Rbb"]
        labels_remapped[(labels >= self.crf_states["Rbb0"]) & (labels <= self.crf_states["Rbb"+str(max_len-1)])] = self.label_to_int["R"]
        max_len = self.max_segment_lengths["Faa"] if self.max_segment_lengths["Faa"] != -1 else self.min_segment_lengths["Faa"]
        labels_remapped[(labels >= self.crf_states["Faa0"]) & (labels <= self.crf_states["Faa"+str(max_len-1)])] = self.label_to_int["F"]
        max_len = self.max_segment_lengths["Fbb"] if self.max_segment_lengths["Fbb"] != -1 else self.min_segment_lengths["Fbb"]
        labels_remapped[(labels >= self.crf_states["Fbb0"]) & (labels <= self.crf_states["Fbb"+str(max_len-1)])] = self.label_to_int["F"]
        
        if isinstance(labels, torch.Tensor):
            assert torch.all(labels_remapped != -1), "Some labels were not converted. This is a bug."
        else:
            assert np.all(labels_remapped != -1), "Some labels were not converted. This is a bug."
        return labels_remapped
    
    def convert_crf_states_to_labels_batch(self, labels: List[List[int]]) -> List[np.ndarray]:
        """
        Convert CRF states back to simple {M,B,R,F,S,>,a,b} labels.
        Applies convert_crf_states_to_labels to each sequence in a batch as returned by the CRF layer.

        Parameters
        ----------
        labels: List[List[int]]
            A list of lists of CRF state integers.
            This is the format returned by the CRF layer.
        
        Returns
        -------
        List[np.ndarray]
            A label array for each sequence in the batch.
        """
        return [self.convert_crf_states_to_labels(np.array(x)) for x in labels]


    def get_sequence_type_from_labels(self, labels: Union[np.ndarray, List[int]]) -> int:
        """
        Infer the sequence type from the label sequence.
        This simply checks for the presence of labels and decides
        by hardcoded rules. As the state space is constrained, this
        should always work, because conflicting labels are not possible.

        This encodes a precedence of labels (compare with header label generation code).
        The "feature precence" is ``S`` > ``>`` > ``R+F`` > ``R`` > ``F`` > ``M``. E.g.
        if an ``M`` sequence ha both ``>`` and ``F`` it is considered ``M+T``, not ``M+F``.

        Parameters
        ----------
        labels: Union[np.ndarray, List[int]]
            A list of labels. This expects the simple {M,B,R,F,S,>,a,b} labels, not CRF states.

        Returns
        -------
        int
            The sequence type, as defined in the self.type_label_to_int dict.

        Raises
        ------
        ValueError
            If the sequence type cannot be inferred.
        """
        if isinstance(labels, list):
            labels = np.array(labels)

        # if there is a B, the sequnence is considered a beta barrel
        if self.label_to_int["B"] in labels:
            return self.type_label_to_int["B"] 
        elif self.label_to_int["M"] in labels:
            if self.label_to_int["S"] in labels:
                return self.type_label_to_int["M+S"]
            elif self.label_to_int[">"] in labels:
                return self.type_label_to_int["M+T"]
            elif self.label_to_int["R"] in labels and self.label_to_int["F"] in labels:
                return self.type_label_to_int["M+R+F"]
            elif self.label_to_int["R"] in labels:
                return self.type_label_to_int["M+R"]
            elif self.label_to_int["F"] in labels:
                return self.type_label_to_int["M+F"]
            else:
                return self.type_label_to_int["M"]

        elif self.label_to_int["S"] in labels:
            return self.type_label_to_int["S"]
        elif self.label_to_int[">"] in labels:
            return self.type_label_to_int["T"]

        elif (self.label_to_int["a_alpha"]== labels).all():
            return self.type_label_to_int["I"]
        elif (self.label_to_int["b_alpha"]== labels).all():
            return self.type_label_to_int["I"]

        # NOTE we can have all-b topologies in the grammar - but 
        # that would only be valid if there were TM segments or a signal peptide.
        # we treat them as globular, as in early training this can and will happen.
        elif (self.label_to_int["a_beta"]== labels).all():
            return self.type_label_to_int["I"]
        elif (self.label_to_int["b_beta"]== labels).all():
            return self.type_label_to_int["I"]
        
        # NOTE we can have R/F without M during training. Consider these as globular.
        elif self.label_to_int["R"] in labels or self.label_to_int["F"] in labels:
            return self.type_label_to_int["I"]

        else:
            raise ValueError(f"Unexpected label - cannot infer type:\n {''.join([self.int_to_label[x] for x in labels])}")


    def get_sequence_type_from_labels_batch(self, labels: List[List[int]]) -> List[int]:
        """
        Infer the sequence type from the label sequence.
        Applies get_sequence_type_from_labels to each sequence in a batch.

        Parameters
        ----------
        labels: List[List[int]]
            A list of lists of labels. This expects the simple {M,B,R,F,S,>,a,b} labels, not CRF states.

        Returns
        -------
        List[int]
            A list of sequence types for each sequence in the batch.
        """
        return [self.get_sequence_type_from_labels(x) for x in labels]

        
            
        


    def tile_emissions(self, emissions: torch.tensor) -> torch.tensor:
        '''
        Broadcast type emissions to all related labels. 
        This is used by the model when passing emissions to the CRF layer.

        Parameters
        ----------
        emissions: torch.tensor
            A 3D tensor of shape (batch_size, sequence_length, len(label_to_int)).
            The emission scores for each label.

        Returns
        -------
        torch.tensor
            A 3D tensor of shape (batch_size, sequence_length, len(CRF_STATES)).
        '''

        emissions_out = torch.zeros(emissions.shape[0], emissions.shape[1], len(self.crf_states), dtype=emissions.dtype, device=emissions.device)

        emissions_out[:,:, [0]] = emissions[:,:,0].unsqueeze(-1) # a_alpha
        emissions_out[:,:, [1]] = emissions[:,:,1].unsqueeze(-1) # b_alpha
        emissions_out[:,:, [2]] = emissions[:,:,2].unsqueeze(-1) # a_beta
        emissions_out[:,:, [3]] = emissions[:,:,3].unsqueeze(-1) # b_beta
        emissions_out[:,:, self.signal_state_inds] = emissions[:,:,4].unsqueeze(-1)
        emissions_out[:,:, self.helix_state_inds] =  emissions[:,:,5].unsqueeze(-1)
        emissions_out[:,:, self.beta_state_inds] =  emissions[:,:,6].unsqueeze(-1)
        emissions_out[:,:, self.reentrant_state_inds] =  emissions[:,:,7].unsqueeze(-1)
        emissions_out[:,:, self.interfacial_state_inds] =  emissions[:,:,8].unsqueeze(-1)
        emissions_out[:,:, self.transit_state_inds] =  emissions[:,:,9].unsqueeze(-1)

        return emissions_out



    def _get_crf_state_labels(self, segment_type: str, length: int) -> np.ndarray:
        '''Get CRF state labels for a segment of the indicated type for the given length.
        
        Parameters
        ----------
        segment_type : str
            A string that denotes the segment type, as used in the CRF state dictionary. Also
            needs to be a valid key in the minumum and maximum length dictionaries.
        length : int
            The length of the segment for which to get the state label sequence.

        Returns
        -------
        np.ndarray
            A numpy array of integers that denote the CRF states for the segment of shape (length,).
        '''
        crf_labels = []
        labels_to_add = length

        # Consider segment_type `S`. 
        # We first add the initial states S0, S1, ... until we reach half the minimum length.
        # Once we have the minimum length, we add remaining positions towards the maximum length,
        # so that the last state is Smax, the preceding one is Smax-1, and so on.

        
        if self.max_segment_lengths[segment_type] == -1:
            # add the min_len states and then add self-loop state.
            for i in range(self.min_segment_lengths[segment_type]):
                crf_labels.append(self.crf_states[segment_type+str(i)])
            for i in range(self.min_segment_lengths[segment_type], length):
                crf_labels.append(self.crf_states[segment_type+str(self.min_segment_lengths[segment_type] - 1)])

        # starting from max_len - remaining positions, add remaining positions towards the maximum length
        else:
            beginning = math.ceil(self.min_segment_lengths[segment_type] / 2)
            # add initial states until we reach half the minimum length
            for i in range(beginning):
                crf_labels.append(self.crf_states[segment_type+str(i)])

            for i in range(self.max_segment_lengths[segment_type] - (labels_to_add - beginning), self.max_segment_lengths[segment_type]):
                crf_labels.append(self.crf_states[segment_type+str(i)])

        return np.array(crf_labels)
    

    def get_label_array(self, topology_str: str) -> np.ndarray:
        """
        Converts string into a numpy array of integers that represent the labels.
        We use this for metrics.
        """

        return np.array([self.label_to_int[x] for x in topology_str])
    
    def get_string_from_labels(self, x: np.ndarray) -> str:
        """
        Converts a numpy array of integers into a string of labels.
        We use this for metrics.
        """
        out = []
        for i in range(len(x)):
            # handle a,b specially - int_to_label has a_alpha, a_beta, b_alpha, b_beta
            if x[i] == 0:
                out.append("a")
            elif x[i] == 1:
                out.append("b")
            else:
                out.append(self.int_to_label[x[i]])

        return ''.join(out)

        # return ''.join([self.int_to_label[x] for x in labels])


    def get_multi_tag_label_matrix(self, topology_str: str, relax: int = 0, relax_globular: bool = False, alphafold_u_mode: bool = True) -> np.ndarray:
        """
        Get a matrix of shape (len(topology_str), len(self.crf_states)) that has a 1 in each position
        where the topology string allows for a segment position of the corresponding type. 
        This can handle unknowns, and will add all possible segments that could fit in the unknown regions.

        Special case handling:

        - If the topology string is all "a", and ``relax_globular`` is True, we also allow for "b" in the matrix, 
            so the whole protein may either be side a or side b (as constrained by the CRF state space).
        - Transit peptides that are annotated with a single ">" at position 0 get a valid TP of any length, up
            to the maximum TP length or the start of the next segment.
        - Samples that are annotated with ">" and "U" (= soluble transit peptide samples) are only allowed to have
            "a" or "b" in the unknown positions (the TP of the exact length is enforced, but we make no statement about
            the orientation of the protein).

        Parameters
        ----------
        topology_str : str
            A string that denotes the topology, with ``U`` for unknown. Valid letters are ``S``, ``>``, ``M``, ``B``, ``R``, ``F``, ``a``, ``b``, ``U``, ``u``, ``.``.
            The lower case ``u`` is used for unknown regions that are not TM/signal. ``.`` is used in beta barrels for not TM (but maybe signal).
        relax : int, optional
            If >0, we "relax" the positions of annotated segments by this amount. This means that we allow the segment to start
            relax positions earlier, and end relax positions later. This can be used if the segment label is not considered to be 
            perfectly accurate, by default 0
        relax_globular : bool, optional
            If True, globular proteins are "relaxed" in the sense that we leave it to the model whether it wants to annotate them as
            side a or side b.

        
        Returns
        -------
        np.ndarray
            A matrix of shape (len(topology_str), len(self.crf_states)) that has a 1 in each position
            where the topology string allows for a segment position of the corresponding type.
        """
        tag_matrix = np.zeros((len(topology_str), len(self.crf_states)))

        # shortcut for all-a (all-b doesn't exist in the data)
        if topology_str == 'a'*len(topology_str):
            # NOTE this case would in principle also be handled by the complete toplogy loop.
            tag_matrix[:, self.crf_states['a_alpha']] = 1
            if relax_globular:
                # But this case needs special handling here.
                tag_matrix[:, self.crf_states['b_alpha']] = 1

            return tag_matrix

        side_a, side_b = ('a_beta', 'b_beta') if "B" in topology_str else ("a_alpha", "b_alpha")

        # a helper function to add the "relaxed" versions of a (start, end, segment_type)
        # segment to the tag matrix. Adds shortened and lengthened versions of label, left
        # and right shifted by i positions.
        # when shifting, we need to make sure we don't violate the start and end positions of
        # the adjacent segments.
        def _add_relaxed_positions(segment_type: str, end: int, start: int, previous_end: int = None, next_start:int = None):
            if previous_end is None:
                previous_end = 0
            if next_start is None:
                next_start = len(topology_str)

            for i in range(1, relax+1):
                labels = self._get_crf_state_labels(segment_type, end-start-i) # shorten by i
                positions = np.arange(start+i, end)
                tag_matrix[positions, labels] = 1
                positions = np.arange(start, end-i)
                tag_matrix[positions, labels] = 1
                
                if not end+i > next_start:
                    labels = self._get_crf_state_labels(segment_type, end-start+i) # lengthen by i
                    positions = np.arange(start, end+i)
                    tag_matrix[positions, labels] = 1
                if not start-i < previous_end:
                    labels = self._get_crf_state_labels(segment_type, end-start+i) # lengthen by i
                    # print(start, end, i, len(labels))
                    positions = np.arange(start-i, end)
                    tag_matrix[positions, labels] = 1

        def find_next_tm_start(current_end):
            '''get the start of the next M,B,R,F segment.'''
            offset_str = topology_str[current_end:]

            next_start = len(topology_str)
            # find returns -1 if not found, so we need to check for that.
            next_start = min(next_start, (current_end + offset_str.find('M')) if offset_str.find('M') != -1 else next_start)
            next_start = min(next_start, (current_end + offset_str.find('B')) if offset_str.find('B') != -1 else next_start)
            next_start = min(next_start, (current_end + offset_str.find('R')) if offset_str.find('R') != -1 else next_start)
            next_start = min(next_start, (current_end + offset_str.find('F')) if offset_str.find('F') != -1 else next_start)
            return next_start



        # turn the label string into a list of unique segments, and their start and end positions
        elements, starts, ends = self._get_unique_consecutive_segments(topology_str)

        if 'M' in topology_str or 'B' in topology_str:
            first_mb_segment_idx = min([i for i, x in enumerate(elements) if x in ['M', 'B', 'R', 'F']])
            last_mb_segment_idx = max([i for i, x in enumerate(elements) if x in ['M', 'B', 'R', 'F']])
            num_B_segments = elements.count('B')

        # shortcut special case for TP+soluble.
        # NOTE TP+TM is handled by the loop below.
        if set(topology_str) == {">", "u"}: # small u only. Large U doesn't shortcut, need to add potential TMs.
            positions = np.arange(starts[0], ends[0])
            labels = self._get_crf_state_labels('>', ends[0]-starts[0])
            tag_matrix[positions, labels] = 1
            # a, b for remaining positions
            tag_matrix[ends[0]:, self.crf_states[side_a]] = 1
            tag_matrix[ends[0]:, self.crf_states[side_b]] = 1
            assert not (tag_matrix.sum(axis=1) == 0).any(), "Labels are missing at positions. This is a bug."
            return tag_matrix


        # these are used by the "relax" function in order to not relax too far into adjacent segments.
        # we update them whenever we process an S,M,B,R,F segment.
        previous_end = None
        next_start = None


        for idx, (element, start, end) in enumerate(zip(elements, starts, ends)):


            # if we have unknown at the start, there can be a signal or transit peptide of any length.
            if element in ['U', '.'] and start == 0 and self.undetermined_mode == 'segments':
                # add signal peptide of any possible length
                for i in range(self.min_segment_lengths['S'], min(self.max_segment_lengths['S'], end)):
                    labels = self._get_crf_state_labels('S', i)
                    positions = np.arange(0, i)
                    tag_matrix[positions, labels] = 1

                for i in range(self.min_segment_lengths['>'], min(self.max_segment_lengths['>'], end)):
                    labels = self._get_crf_state_labels('>', i)
                    positions = np.arange(0, i)
                    tag_matrix[positions, labels] = 1

                # we do not enforce start-end transitions - this means we can also have "incomplete" signal peptides and TPs.
                if not self.use_start_end_transitions:
                    # get a maximum length SP
                    labels = self._get_crf_state_labels('S', self.max_segment_lengths['S'])
                    # insert all labels[i:] truncations that fit
                    for i in range(1, min(end, self.max_segment_lengths['S'])):
                        truncated_label = labels[self.max_segment_lengths['S']-i:] # get the last i labels
                        positions = np.arange(0, len(truncated_label))
                        tag_matrix[positions, truncated_label] = 1

                    labels = self._get_crf_state_labels('>', self.max_segment_lengths['>'])
                    # insert all labels[i:] truncations that fit
                    for i in range(1, min(end, self.max_segment_lengths['>'])):
                        truncated_label = labels[self.max_segment_lengths['>']-i:]
                        positions = np.arange(0, len(truncated_label))
                        tag_matrix[positions, truncated_label] = 1

            elif element == 'U' and start == 0 and self.undetermined_mode == 'any':
                # just set all states to true.
                tag_matrix[:end, :] = 1
                        
                        
            # NOTE the above is special and we still want to loop through the options below if it's true in order to add the remaining options implied by U.

            if element == 'S':
                # special case for len=1 SP annotation - this implies any-length.
                if end-start == 1:
                    max_tp_pos = min(self.max_segment_lengths['S'], find_next_tm_start(end)) if self.max_segment_lengths['S'] != -1 else find_next_tm_start(end)
                    for i in range(self.min_segment_lengths['S'], max_tp_pos):
                        labels = self._get_crf_state_labels('S', i)
                        positions = np.arange(start, start+i)
                        tag_matrix[positions, labels] = 1
                
                else:
                    positions = np.arange(start, end)
                    labels = self._get_crf_state_labels('S', end-start)
                    tag_matrix[positions, labels] = 1

                previous_end = end

            elif element == '>':
                # special case for len=1 TP annotation - this implies any-length.
                if end-start == 1:
                    max_tp_pos = min(self.max_segment_lengths['>'], find_next_tm_start(end)) if self.max_segment_lengths['>'] != -1 else find_next_tm_start(end)
                    for i in range(self.min_segment_lengths['>'], max_tp_pos):
                        labels = self._get_crf_state_labels('>', i)
                        positions = np.arange(start, start+i)
                        tag_matrix[positions, labels] = 1


                else:
                    positions = np.arange(start, end)
                    labels = self._get_crf_state_labels('>', end-start)
                    tag_matrix[positions, labels] = 1

                previous_end = end

            # If the element is a known state, add the corresponding CRF states
            elif element == 'a':
                positions = np.arange(start, end)
                tag_matrix[positions, self.crf_states[side_a]] = 1
            elif element == 'b':
                positions = np.arange(start, end)
                tag_matrix[positions, self.crf_states[side_b]] = 1

            # If the element is a known state, add the corresponding CRF states
            elif element in ['M', 'B'] :
                positions = np.arange(start, end)
                next_start = find_next_tm_start(end)

                # if the previous element is unknown, the TM segment can be into any direction
                if elements[idx-1] in ['U', 'u', '.']:
                    segment_type = element + 'ab'
                    labels = self._get_crf_state_labels(segment_type, end-start)
                    tag_matrix[positions, labels] = 1

                    if relax>0:
                        _add_relaxed_positions(segment_type, end, start, previous_end, next_start)

                    segment_type = element + 'ba'
                    labels = self._get_crf_state_labels(segment_type, end-start)
                    tag_matrix[positions, labels] = 1

                    if relax>0:
                        _add_relaxed_positions(segment_type, end, start, previous_end, next_start)

                # if the previous element is known, the TM segment can only be in the direction out of the previous element
                elif elements[idx-1] in ['a', 'b']:
                    segment_type = element + 'ab' if elements[idx-1] == 'a' else element + 'ba'
                    labels = self._get_crf_state_labels(segment_type, end-start)
                    tag_matrix[positions, labels] = 1

                    if relax>0:
                        _add_relaxed_positions(segment_type, end, start, previous_end, next_start)

                else :
                    raise ValueError(f'Unknown element in topology string: {elements[idx-1]} transitions to {element}')
                
                previous_end = end


            # If the element is a known state, add the corresponding CRF states
            elif element in ['F', 'R'] :
                positions = np.arange(start, end)
                next_start = find_next_tm_start(end)

                # if the previous element is unknown, the TM segment can be into any direction
                if elements[idx-1] in ['U', 'u', '.']:
                    segment_type = element + 'aa'
                    labels = self._get_crf_state_labels(segment_type, end-start)
                    tag_matrix[positions, labels] = 1

                    if relax>0:
                        _add_relaxed_positions(segment_type, end, start, previous_end, next_start)

                    segment_type = element + 'bb'
                    labels = self._get_crf_state_labels(segment_type, end-start)
                    tag_matrix[positions, labels] = 1

                    if relax>0:
                        _add_relaxed_positions(segment_type, end, start, previous_end, next_start)

                # if the previous element is known, the TM segment can only be in the direction back to the previous element
                elif elements[idx-1] in ['a', 'b']:
                    segment_type = element + 'aa' if elements[idx-1] == 'a' else element + 'bb'
                    labels = self._get_crf_state_labels(segment_type, end-start)
                    tag_matrix[positions, labels] = 1

                    if relax>0:
                        _add_relaxed_positions(segment_type, end, start, previous_end, next_start)

                else :
                    raise ValueError(f'Unknown element in topology string: {elements[idx-1]} transitions to {element}')
                
                previous_end = end
                

            # the general unknown case, where anything is possible.
            # positions could be a,b, or contain a M,B,F,R of any length at any position.
            elif element in ['U', 'u', '.']:

                if self.undetermined_mode == 'segments':
                    positions = np.arange(start, end)

                    tag_matrix[positions, self.crf_states[side_a]] = 1
                    tag_matrix[positions, self.crf_states[side_b]] = 1


                    # add all possible TM segments that would fit into this unknown region.
                    if 'B' in topology_str:
                        if element in ['u', '.']: # SP or a/b. SP already handled before.
                            allowed = []
                        else:
                            allowed = ['Bba', 'Bab'] 
                    else: 
                        if element == 'U':
                            allowed = ['Mab', 'Mba', 'Faa', 'Fbb', 'Raa', 'Rbb']
                        else:
                            # lower case u - not TM.
                            allowed = ['Faa', 'Fbb', 'Raa', 'Rbb']

                    # if this is between the first and last TM segment and U, for alphafold
                    # samples we assume that there cannot be any extra TM regions in-between the 
                    # labeled TM regions. This is probably not true altogether, but it's a simplification
                    # that could give us more training signal from these samples.
                    # if alphafold_u_mode and (idx > first_mb_segment_idx and idx < last_mb_segment_idx):
                    #     allowed = allowed[2:] # only allow globular/R/F segments in-between TM segments.

                    # NOTE assume if we have 8 or more B segments, the barrel is complete.
                    # before we had that as 1 B, likely dumb.
                    if alphafold_u_mode and num_B_segments >= 8:
                    # if alphafold_u_mode and "B" in topology_str:
                        allowed = []
                        

                    for maybe_start in range(start, end):
                        for maybe_end in range(start, end):
                            length = maybe_end - maybe_start
                            positions = np.arange(maybe_start, maybe_end)
                            for k in allowed:
                                if length >= self.min_segment_lengths[k] and length <= self.max_segment_lengths[k]:
                                    labels = self._get_crf_state_labels(k, length)
                                    tag_matrix[positions, labels] = 1


                    # special N terminus case if we don't use start-end transitions
                    if not self.use_start_end_transitions and idx == 0:
                        # add all possible N-truncated TM segments that would fit into this unknown region.
                        for k in allowed:
                            # get maximum length segment
                            labels = self._get_crf_state_labels(k, self.max_segment_lengths[k])
                            # insert all labels[i:] truncations that fit
                            for i in range(1, min(end, self.max_segment_lengths[k])):
                                truncated_label = labels[self.max_segment_lengths[k]-i:] # get the last i labels
                                positions = np.arange(0, len(truncated_label))
                                tag_matrix[positions, truncated_label] = 1

                    # special C terminus case if we don't use start-end transitions
                    if not self.use_start_end_transitions and idx == len(elements)-1:
                        # add all possible C-truncated TM segments that would fit into this unknown region.
                        for k in allowed:
                            # get maximum length segment
                            labels = self._get_crf_state_labels(k, self.max_segment_lengths[k])
                            # insert all labels[:i] truncations that fit
                            for i in range(1, min(end-start, self.max_segment_lengths[k])):
                                truncated_label = labels[:i]
                                positions = np.arange(end-i, end)
                                assert len(positions) == len(truncated_label)
                                tag_matrix[positions, truncated_label] = 1

                elif self.undetermined_mode == 'any':
                    if alphafold_u_mode and (idx > first_mb_segment_idx and idx < last_mb_segment_idx):
                        # only allow a,b,R,F in-between TM segments.
                        tag_matrix[start:end, self.crf_states[side_a]] = 1
                        tag_matrix[start:end, self.crf_states[side_b]] = 1
                        if not "B" in topology_str:
                            tag_matrix[start:end, self.reentrant_state_inds] = 1
                            tag_matrix[start:end, self.interfacial_state_inds] = 1

                    else:
                        positions = np.arange(start, end)
                        tag_matrix[positions, :] = 1



            else :
                raise ValueError(f'Unknown element in topology string: {element}')
                
        if (tag_matrix.sum(axis=1) == 0).any():
            raise ValueError(f'Undefined positions in label matrix (=correct processing failed) for string: {topology_str}')

        return tag_matrix


    def validate_labels(self, labels: str):
        '''A helper function to validate that labels do not violate the state space.
        
        Parameters
        ----------
        labels : str
            A string of labels that represent the topology of a protein sequence.

        Raises
        ------
        ValueError
            If the labels violate the state space.
        '''

        elements, starts, ends = self._get_unique_consecutive_segments(list(labels))

        for idx, (element, s, e) in enumerate(zip(elements, starts, ends)):
            
            # we only need to check the segments of interest.
            if element in ['U', 'u', '.', 'a', 'b']:
                continue

            # len-1 TPs followed by U are allowed.
            if element == '>' and elements[idx+1] == 'U':
                continue
            if element == 'S' and elements[idx+1] in ['U', '.']:
                continue

            if e-s < self.min_segment_lengths[element]:
                raise ValueError(f'Segment {element} at {s}-{e} too short: {e-s}<{self.min_segment_lengths[element]}')
            if e-s > self.max_segment_lengths[element] and self.max_segment_lengths[element] != -1:
                raise ValueError(f'Segment {element} at {s}-{e} too long: {e-s}>{self.max_segment_lengths[element]}')

        # detect I-M-I, O-M-O
        for i in range(1, len(elements)-1):

            if elements[i] in ['M', 'B']:
                if elements[i-1] == 'a' and elements[i+1] == 'a':
                    raise ValueError(f'a-a transition found at segment {i}')
                if elements[i-1] == 'b' and elements[i+1] == 'b':
                    raise ValueError(f'b-b transition found at segment {i}')
                

            elif elements[i] in ['F', 'R']:
                if elements[i-1] == 'a' and elements[i+1] == 'b':
                    raise ValueError(f'a-b transition found at segment {i}')
                if elements[i-1] == 'b' and elements[i+1] == 'a':
                    raise ValueError(f'b-a transition found at segment {i}')
                

    def _validate(self):
        """
        Validate that the CRF config is valid.

        We override this method from the base class, as we use the double a/b mapping.
        """
        # validate that ints are unique
        # assert len(set(self.label_to_int.values())) == len(self.label_to_int.values())
        pass