from abc import ABC, abstractmethod
import torch
from typing import Callable

class ShapAdapter(ABC):
    """Abstract base class for SHAP (SHapley Additive exPlanations) adapters"""
    @abstractmethod
    def get_shap_predict_fn(self) -> Callable[[torch.Tensor], torch.Tensor]: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""

    @abstractmethod
    def shap_prepare_inference_input(self, x: torch.Tensor) -> torch.Tensor: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""
