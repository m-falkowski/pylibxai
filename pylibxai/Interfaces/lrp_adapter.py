from abc import ABC, abstractmethod
import torch.nn as nn

class LrpAdapter(ABC):
    """Abstract base class for LRP (Layer-wise Relevance Propagation) adapters"""
    @abstractmethod
    def get_lrp_predict_fn(self) -> nn.Module: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""