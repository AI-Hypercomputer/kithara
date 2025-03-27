from typing import Protocol, Any
from kithara.callbacks import Checkpointer 
from abc import ABC, abstractmethod
class RayModel(ABC):
    """Protocol defining the interface for ray-based models."""
    @property
    def model(self) -> float | None:
        ...
    @property
    def checkpointer(self) -> Checkpointer | None:
        ...
    def save_checkpoint(self, step):
        self.checkpointer.save(step)
