import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
        
def enable_dropout(model: nn.Module):
    """Set all Dropout layers to training mode to enable stochastic forward passes."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class LayerInputExtractor:
    """Forward-hook wrapper that captures the input to a given layer on each forward pass."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer
        self._features = None
        self.hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        self._features = inputs[0].detach()

    def __call__(self, x):
        _ = self.model(x)
        return self._features

    def close(self):
        self.hook.remove()

def get_last_layer(model: nn.Module,
                   dataloader: DataLoader,
                   device: torch.device):
    """Yield averaged final-block input features for each batch of sequences."""
    model.to(device).eval()
    extractor = LayerInputExtractor(model, model.final_block)
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)
            result = extractor(X)
            result = result.reshape((result.shape[0], -1))
            half_batch = result.shape[0] // 2
            yield (result[:half_batch, :] + result[half_batch:, :]) / 2

def distance_np(target: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.sum((target-X)**2,axis=1)

def distance_torch(target: torch.Tensor,X: torch.Tensor) -> torch.Tensor:
    return torch.sum((target-X)**2,dim=1)
