
'''
The configuration of the CRF state space.
'''

import math
import numpy as np
import torch
from typing import Dict, Union, List, Any, Tuple
from itertools import groupby



# # Segment lengths
# # max_signal_length = 60
# # max_alpha_membrane_length = 30
# # max_beta_membrane_length = 15
# # max_
# MINIMUM_SEGMENT_LENGTHS = {
#     'S': 10,   # Signal
#     '>': 10, # Transit peptide
#     'M': 10,
#     'B': 5,
#     'R': 8,
#     'F': 6,
#     'Mab': 10, # Alpha TM
#     'Mba': 10, # Alpha TM
#     'Bab': 5,  # Beta TM
#     'Bba': 5,  # Beta TM
#     'Raa': 8, # Reentrant
#     'Rbb': 8, # Reentrant
#     'Faa': 6, # Interfacial
#     'Fbb': 6, # Interfacial
# }

# MAXIMUM_SEGMENT_LENGTHS = {
#     'S': 60,   # Signal
#     '>': 170, # Transit peptide
#     'M': 30,
#     'B': 15,
#     'R': 30,
#     'F': -1,
#     'Mab': 30, # Alpha TM
#     'Mba': 30, # Alpha TM
#     'Bab': 15,  # Beta TM
#     'Bba': 15,  # Beta TM
#     'Raa': 30, # Reentrant
#     'Rbb': 30, # Reentrant
#     'Faa': -1, # Interfacial
#     'Fbb': -1, # Interfacial
# }




class SplitIOCRFwithTPConfig:
    def __init__(
            self,
            maximum_segment_lengths: Dict[str, int] = None,
            minimum_segment_lengths: Dict[str, int] = None,
            use_start_end_transitions: bool = True,
            type_label_to_int: Dict[str, int] = None,
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

    def simplify_marginals(self, marginals: torch.Tensor) -> Dict[int, np.ndarray]:
        """
        Aggregates raw CRF marginal probabilities into simplified categories for plotting.
        """
        from ..constants import MARGINAL_CATEGORY_MAPPING

        category_probs = {}
        for state_idx, simple_label_idx in self.simplification_dict.items():
            if state_idx >= marginals.shape[1]:
                continue

            plot_cat_idx = MARGINAL_CATEGORY_MAPPING.get(simple_label_idx)

            if plot_cat_idx is None:
                continue
                
            probs = marginals[:, state_idx].cpu().numpy()

            if plot_cat_idx not in category_probs:
                category_probs[plot_cat_idx] = probs
            else:
                category_probs[plot_cat_idx] += probs
                
        return category_probs