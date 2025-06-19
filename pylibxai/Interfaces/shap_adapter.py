from abc import ABC, abstractmethod
import numpy as np

class ShapAdapter(ABC):
    """Abstract base class for SHAP (SHapley Additive exPlanations) adapters"""
    @abstractmethod
    def get_shap_predict_fn(self) -> np.array: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""
