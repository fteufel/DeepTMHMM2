
"""
attention_pooling.py
====================

This module implements the attention pooling layer. The
model was first proposed in DeepLoc 2.0 .
"""
import torch
import torch.nn as nn

class AttentionPoolingModel(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            mlp_dim: int, 
            output_dim: int, 
            dropout: float=0.1
            ):
        """
        Attention pooling along the sequence dimension. 
        This is the core of the DeepLoc 2.0 model.


        Parameters
        ----------
        input_dim : int
            The input dimension of the model.
        mlp_dim : int
            The dimension of the MLP.
        output_dim : int
            The output dimension of the model.
        dropout : float, optional
            The dropout rate, by default 0.1
        """
        super().__init__()

        self.initial_ln = nn.LayerNorm(input_dim)

        self.attention_net = nn.Linear(input_dim, 1)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim)
        )

        self.drop_layer = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Shape: (batch_size, seq_len, input_dim)
        mask : torch.Tensor
            The mask tensor. Shape: (batch_size, seq_len). 1 for valid positions, 0 for masked positions.

        Returns
        -------
        torch.Tensor
            The output tensor. Shape: (batch_size, output_dim)
        """
        x = self.initial_ln(x)

        x = self.drop_layer(x)

        # compute the attention weight of each sequence position
        attention_weights = self.attention_net(x) # (batch_size, seq_len, 1) // float16 to float 32

        # set masked positions to a large negative number so that their softmax will be 0
        #unsqueeze(-1) adds a new dimension after the last index
        attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e4)

        # softmax, make weights sum to 1 for each sequence
        attention_weights = nn.functional.softmax(attention_weights, dim=1) # (batch_size, seq_len, 1) #dim=1 rows sum 1

        # weighted sum
        x_pooled = (x * attention_weights).sum(dim=1) # (batch_size, embedding_dim)

        # predict from sequence embeddings
        logits = self.mlp(x_pooled)
        return torch.sigmoid(logits)
