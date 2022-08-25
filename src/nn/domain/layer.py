import numpy as np

from .activation import Activation


class Layer:
    def __init__(self, shape, activation: Activation):
        self.n_inputs = shape[1]
        self.n_neurons = shape[0]
        self.weights = 0.1 * np.random.randn(shape[0], shape[1] + 1)
        self.activation = activation

    def forward(self, input):
        input_with_extra_row = np.r_[input, np.ones(1)]
        return self.activation.eval(np.dot(self.weights, input_with_extra_row))

    # def back_propagation(self, cost_wrt_values, prev_values, learning_rate):
    #    """ ToDo """
    #    cost_wrt_weights = self.values_wrt_weights() * cost_wrt_values
    #    self.weights = np.subtract(self.weights, learning_rate * cost_wrt_weights)
    #    cost_wrt_values = np.dot(self.values_wrt_previous_values(prev_values), cost_wrt_values)
    #    return cost_wrt_values

    def values_wrt_weights(self, prev_values):
        return []

    def values_wrt_previous_values(self, prev_values):
        return []

    def __str__(self):
        return (
            "\n"
            + "-" * 65
            + f"\nInputs: {self.n_inputs}\nNeurons: {self.n_neurons}\n"
            + f"Activation: {self.activation}\nWeights:\n"
            + f"{self.weights[:-1]}\nBias:\n{self.weights[-1]}\n"
            + "-" * 65
        )
