"""
This module implements a basic Neural Network layer.
"""
import numpy as np

from .activation import Activation


class Layer:
    """
    This class implements a basic Neural Network layer.
    """

    def __init__(self, shape, activation: Activation):
        self.n_inputs = shape[1]
        self.n_neurons = shape[0]
        self.activation = activation
        self.reset_weights()

    def reset_weights(self):
        """
        Generates random values for all the layer's weights.
        """
        self.weights = 0.1 * np.random.randn(self.n_neurons, self.n_inputs)
        self.bias = 0.1 * np.random.randn(self.n_neurons, 1)

    def __str__(self):
        return (
            "\n"
            + "-" * 65
            + f"\nInputs: {self.n_inputs}\nNeurons: {self.n_neurons}\n"
            + f"Activation: {self.activation}\nWeights:\n"
            + f"{self.weights}\nBias:\n{self.bias}\n"
            + "-" * 65
        )
