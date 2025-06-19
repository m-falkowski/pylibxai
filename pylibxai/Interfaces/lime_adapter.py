from abc import ABC, abstractmethod
import numpy as np

class LimeAdapter(ABC):
    """Abstract base class for LIME (Local Interpretable Model-agnostic Explanations) adapters"""
    @abstractmethod
    def get_lime_predict_fn(self) -> np.array: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""