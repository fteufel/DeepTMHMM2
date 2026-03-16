import torch
from typing import List, Tuple, Optional
from .lightning_deeptmhmm import DeepTMLightning
from .lightning_memtype import MembraneTypeLightning

class ModelContainer:
    """Wraps multiple topology and membrane type models for ensemble prediction."""
    def __init__(self, topology_checkpoints: List[str], memtype_checkpoints: List[str], device: str):
        self.topology_models = []
        for p in topology_checkpoints:
            print(f"Loading topology model from {p}...")
            lightning_model = DeepTMLightning.load_from_checkpoint(p, strict=False, map_location=torch.device(device), weights_only=False)
            self.topology_models.append(lightning_model.model)

        self.memtype_models = []
        for p in memtype_checkpoints:
            print(f"Loading membrane type model from {p}...")
            lightning_model = MembraneTypeLightning.load_from_checkpoint(p, strict=False, map_location=torch.device(device), weights_only=False)
            self.memtype_models.append(lightning_model.model)

        self.device = device
        self.to(self.device)
        for model in self.topology_models:
            model.predict_marginals = True
            model.eval()

        for model in self.memtype_models:
            model.eval()

    def to(self, device: str) -> "ModelContainer":
        for model in self.topology_models:
            model.to(device)
        for model in self.memtype_models:
            model.to(device)
        return self

    def predict_single_model(self, model_idx: int, embedding: torch.Tensor, mask: torch.Tensor, compute_marginals: bool=False) -> Tuple[str, torch.Tensor, dict, float]:
        with torch.no_grad():
            top_model = self.topology_models[model_idx]
            mem_model = self.memtype_models[model_idx]

            (simple_paths, marginals) = top_model(embedding, mask, compute_marginals=compute_marginals)
            topology_str = top_model.crf_config.get_string_from_labels(simple_paths[0])
            
            simplified_marginals = top_model.crf_config.simplify_marginals(marginals[0]) if marginals is not None else None
            
            mem_type_probs = mem_model(embedding, mask)
            
        return topology_str, mem_type_probs, simplified_marginals, 0.0
