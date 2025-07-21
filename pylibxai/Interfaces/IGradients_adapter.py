from abc import ABC, abstractmethod
import torch
from typing import Callable

class IGradientsAdapter(ABC):
    """Abstract base class for Integrated gradients method adapters"""
    @abstractmethod
    def get_igrad_predict_fn(self) -> Callable[[torch.Tensor], torch.Tensor]: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""

    @abstractmethod
    def igrad_prepare_inference_input(self, x: torch.Tensor) -> torch.Tensor: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""
