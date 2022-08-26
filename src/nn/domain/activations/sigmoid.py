import numpy as np

from ..activation import Activation


class Sigmoid(Activation):
    def __init__(self):
        super().__init__("Sigmoid")

    def eval_element(self, value):
        return 1.0 / (1.0 + np.exp(-value))

    def derivative_element(self, value):
        return self.eval_element(value) * (1 - self.eval_element(value))
