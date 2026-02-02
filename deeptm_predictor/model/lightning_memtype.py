
from typing import Dict
import lightning as L
from .torch_memtype import AttentionPoolingModel

class MembraneTypeLightning(L.LightningModule):
    def __init__(self,
                 embedding_dim: int,
                 dropout_em: int,
                 clf_hidden_dim: int,
                 num_membrane_classes: int,
                 learning_rate=1e-3,
                 membrane_class_names: Dict[int, str | int] | None = None,
                 memtype_is_multilabel: bool = False,
                ):
        super().__init__()

        self.save_hyperparameters()
        self.model = AttentionPoolingModel(
            input_dim=embedding_dim,
            mlp_dim=clf_hidden_dim,
            output_dim=num_membrane_classes,
            dropout=dropout_em
        )
