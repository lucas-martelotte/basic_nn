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
        self.weights = 0.1 * np.random.randn(shape[0], shape[1] + 1)
        self.activation = activation

    def forward(self, input_vector: np.ndarray):
        """
        This class implements a forward iteration of the layer. It does NOT apply the
        activation function yet, only the multiplication by the weights.

        :param input_vector: The input vector. The number 1 will be appended at the end,
            in order to acomodate the bias term.

        :returns: The resulting array after multiplying the input by the weights.
        """
        input_with_extra_row = np.r_[input_vector, np.ones(1)]
        return np.dot(self.weights, input_with_extra_row)

    def __str__(self):
        return (
            "\n"
            + "-" * 65
            + f"\nInputs: {self.n_inputs}\nNeurons: {self.n_neurons}\n"
            + f"Activation: {self.activation}\nWeights:\n"
            + f"{self.weights[:-1]}\nBias:\n{self.weights[-1]}\n"
            + "-" * 65
        )
