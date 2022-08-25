from abc import ABC, abstractmethod
import numpy as np


class Cost(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def eval(self, predicted_output: np.ndarray, sample_output: np.ndarray):
        pass

    @abstractmethod
    def gradient(self, predicted_output: np.ndarray, sample_output: np.ndarray):
        pass

    def __str__(self):
        return self.name
