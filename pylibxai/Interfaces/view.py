from abc import ABC, abstractmethod
from enum import IntEnum 

class ViewType(IntEnum):
    """Enum-like class for View Types"""
    WEBVIEW = 1
    DEBUG = 2
    NONE = 2

    @classmethod
    def values(cls):
        return [cls.CONSOLE, cls.FILE, cls.GRAPHICAL]

class ViewInterface(ABC):
    """Abstract base class for View Interface
    """
    @abstractmethod
    def __init__(self, context):
        self.context = context

    @abstractmethod
    def start(self) -> None: pass

    @abstractmethod
    def stop(self) -> None: pass