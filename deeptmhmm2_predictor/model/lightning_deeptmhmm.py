
from typing import Dict

import lightning as L
from .torch_deeptmhmm import SP5DeepTMHMM

class DeepTMLightning(L.LightningModule):
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 dropout_em: int,
                 dropout_attn: int,
                 dropout_conv1: int,
                 dropout_conv2: int,
                 n_filters: int,
                 filter_size_in: int,
                 filter_size_out: int,
                 num_layers: int,
                 membrane_head_inputs: str,
                 clf_hidden_dim: int,
                 num_membrane_classes: int,
                 accuracy_min_segment_overlaps: Dict[int, int],
                 model_mode: str = 'sp5',
                 learning_rate=1e-3,
                 clf_loss_scale=1.0,
                 membrane_class_names: Dict[int, str | int] | None = None,
                 memtype_is_multilabel: bool = False,
                 **kwargs,
                ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SP5DeepTMHMM(
            embedding_size=embedding_dim, 
            lstm_hidden_size=hidden_dim,
            clf_hidden_size=clf_hidden_dim,
            num_membrane_classes=num_membrane_classes,
            dropout_em=dropout_em,
            dropout_attn=dropout_attn,
            dropout_conv1=dropout_conv1,
            dropout_conv2=dropout_conv2,
            membrane_head_inputs=membrane_head_inputs,
            n_filters=n_filters,
            filter_size_in=filter_size_in,
            filter_size_out=filter_size_out,
            )
