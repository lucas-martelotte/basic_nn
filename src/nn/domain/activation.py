from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    def __init__(self, name):
        self.name = name
        self.eval = np.vectorize(self.eval_element)
        self.derivative_eval = np.vectorize(self.derivative_element)

    @abstractmethod
    def eval_element(self, input):
        pass

    @abstractmethod
    def derivative_element(self, input):
        pass

    def __str__(self):
        return self.name
