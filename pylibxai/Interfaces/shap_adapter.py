from abc import ABC, abstractmethod
import torch
import numpy as np

class ShapAdapter(ABC):
    """Abstract base class for SHAP (SHapley Additive exPlanations) adapters"""
    @abstractmethod
    def get_shap_predict_fn(self) -> np.array: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""

    @abstractmethod
    def shap_prepare_inference_input(self, x: torch.Tensor) -> torch.Tensor: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""

    @abstractmethod
    def shap_map_target_to_id(self, target: str) -> int: pass
    """Maps a target label to its corresponding ID."""
