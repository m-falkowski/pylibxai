from abc import ABC, abstractmethod
import numpy as np
from typing import Callable

class LimeAdapter(ABC):
    """Abstract base class for LIME (Local Interpretable Model-agnostic Explanations) adapters"""
    @abstractmethod
    def get_lime_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]: pass
    """Returns a function that takes an audio input and returns the model's prediction for that input."""