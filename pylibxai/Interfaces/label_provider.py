from abc import ABC, abstractmethod
from typing import Dict

class ModelLabelProvider(ABC):
    """Abstract base class for providing labels for model predictions"""
    @abstractmethod
    def get_label_mapping(self) -> Dict[int, str]: pass
    """Returns a mapping from label IDs to label names."""