
"""
This is largely based on the DeepTMHMM model architecture.
It additionally uses convolutions like SignalP5.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from .crf_config_with_tp_split_io import SplitIOCRFwithTPConfig
from ..constants import CRF_CONFIG
from .crf import CRF


class SP5DeepTMHMM(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 lstm_hidden_size: int,
                 clf_hidden_size: int,
                 dropout_em: float,
                 dropout_conv1: float,
                 dropout_conv2: float,
                 n_filters = 512,#32,
                 filter_size_in = 3,
                 filter_size_out = 5,
                 extend_first_tm: bool = False,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        embedding_size : int
            Size of the input embeddings.
        lstm_hidden_size : int
            Size of the LSTM hidden layer.
        clf_hidden_size : int
            Size of the classifier hidden layer.
        dropout_em : float
            Dropout probability of the emissions (lstm output)
        dropout_attn : float
            Dropout probability of the attention pooling input for
            the membrane type classifier.
        dropout_conv1: float
            Dropout probability of the first convolutional layer.
        dropout_conv2: float
            Dropout probability of the second convolutional layer.
        num_membrane_classes : int
            Number of membrane types.
        membrane_head_inputs : str, optional
            Inputs for the membrane type classifier. Can be 'embedding', 'lstm' or 'both'.
            By default 'embedding'.
        n_filters : int, optional
            Number of filters for the convolutional layers, by default 32.
        filter_size_in : int, optional
            Filter size for the first convolutional layer, by default 3.
        filter_size_out : int, optional
            Filter size for the second convolutional layer, by default 5.
        """
        super(SP5DeepTMHMM, self).__init__()

        crf_config = SplitIOCRFwithTPConfig(**CRF_CONFIG, extend_first_tm=extend_first_tm)
        self.crf_config = crf_config

        # configure NN

        # initialize model variables
        num_tags = len(crf_config.crf_states)
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.clf_hidden_size = clf_hidden_size

        if self.lstm_hidden_size % 2 != 0:
            raise ValueError(f"lstm_hidden_size must be divisible by 2 (bidirectional LSTM), got {self.lstm_hidden_size}")

        self.conv1 = nn.Conv1d(in_channels=embedding_size, out_channels=n_filters,
                            kernel_size=filter_size_in, stride=1, padding=filter_size_in // 2)  # in:20, out=32
        self.bi_lstm = nn.LSTM(n_filters,
                               self.lstm_hidden_size//2,
                               num_layers=1,
                               bidirectional=True,
                               batch_first=True)
        self.conv2 = nn.Conv1d(in_channels=self.lstm_hidden_size, out_channels=n_filters * 2, kernel_size=filter_size_out,
                            stride=1, padding=filter_size_out // 2)  # (128,64)


        self.input_dropout = nn.Dropout(p=dropout_em)
        self.conv1_dropout = nn.Dropout(p=dropout_conv1)
        self.conv2_dropout = nn.Dropout(p=dropout_conv2)

        self.hidden_to_labels = nn.Linear(self.lstm_hidden_size, crf_config.num_emissions)


        # configure CRF - predicting topology.
        self.allowed_transitions = crf_config.allowed_state_transitions
        self.allowed_start_states = crf_config.allowed_start_states
        self.allowed_end_states = crf_config.allowed_end_states

        self.crf = CRF(
            num_tags, 
            batch_first = True,
            init_weight = 0.1, 
            allowed_transitions = crf_config.allowed_state_transitions, 
            allowed_start = crf_config.allowed_start_states, 
            allowed_end = crf_config.allowed_end_states,
            include_start_end_transitions = crf_config.use_start_end_transitions,
            constrain_every = True)
        
        self.simplify_labels = crf_config.convert_crf_states_to_labels_batch
        self._tile_emissions = crf_config.tile_emissions

        self.flatten_parameters()

    def flatten_parameters(self):
        self.bi_lstm.flatten_parameters()



    def _get_network_emissions(self, embeddings, mask, return_lstm_out: bool = False):
        """Run LSTM and broadcast emissions.

        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings of the input sequences.
        mask : torch.Tensor
            Mask of the input sequences indicating which positions are valid.
        return_lstm_out : bool, optional
            Whether to return the LSTM output, by default False.

        Returns
        -------
        torch.Tensor
            Emissions of the CRF.
        torch.Tensor, optional
            LSTM output. Only if return_lstm_out is True.
        """
        x = embeddings
        x = self.input_dropout(x)
        x = nn.functional.relu(self.conv1(x.permute(0, 2, 1))).permute(0, 2, 1)
        x = self.conv1_dropout(x)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(x, mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False)
        bi_lstm_out_packed, _ = self.bi_lstm(packed_embeddings)
        bi_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(bi_lstm_out_packed, batch_first=True)
        x = nn.functional.relu(self.conv2(bi_lstm_out.permute(0, 2, 1))).permute(0, 2, 1)

        bi_lstm_out = self.conv2_dropout(bi_lstm_out)
        emissions = self.hidden_to_labels(bi_lstm_out)
        emissions = self._tile_emissions(emissions)

        if return_lstm_out:
            return emissions, bi_lstm_out
        else:
            return emissions


    def forward(
            self, 
            embeddings: torch.Tensor, 
            mask: torch.Tensor, 
            compute_marginals: bool = False,
            ) -> Tuple[List[np.ndarray], List[int], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        emissions, bi_lstm_out = self._get_network_emissions(embeddings, mask, return_lstm_out=True)

        paths = self.crf.decode(emissions, mask.byte()) # List[List[int]]
        simple_paths = self.simplify_labels(paths)

        if compute_marginals:
            marginals = self.crf.compute_marginal_probabilities(emissions, mask, truncate=False)
        else:
            marginals = None


        outs = (simple_paths, marginals)
        return outs