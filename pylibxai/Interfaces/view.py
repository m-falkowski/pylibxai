from abc import ABC, abstractmethod

class ViewInterface(ABC):
    """Abstract base class for View Interface
    """
    @abstractmethod
    def __init__(self, directory, port):
        self.directory = directory
        self.port = port

    @abstractmethod
    def start(self) -> None: pass

    @abstractmethod
    def stop(self) -> None: pass