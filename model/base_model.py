"""
Abstract class for base model
"""

from abc import ABC, abstractmethod
from configs import config


class BaseModel(ABC):
    """Abstract Model class that is inherit to all models"""

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
