from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    def __init__(self, name):
        self.name = name
        self.eval = np.vectorize(self.eval_element)
        self.derivative = np.vectorize(self.derivative_element)

    @abstractmethod
    def eval_element(self, value):
        pass

    @abstractmethod
    def derivative_element(self, value):
        pass

    def __str__(self):
        return self.name
