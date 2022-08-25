"""
    This module tests the funcionality of the Layer class.
"""
import numpy as np

from src.nn import Layer, ReLu


def test_layer_constructor():
    """
    The Layer constructor should create the layer correctly.
    """
    layer = Layer((4, 3), ReLu())
    assert isinstance(layer.activation, ReLu)
    assert layer.n_inputs == 3
    assert layer.n_neurons == 4
    assert np.shape(layer.weights) == (4, 4)


def test_layer_forward():
    """
    Layer should only multiply the input by
    the weights and add the bias when forward is called.
    """
    layer = Layer((2, 3), ReLu())
    layer.weights = np.array([[1, 2, 3, 4], [3, -2, 2, 1]])
    input_vector = np.array([1, 5, -2])
    expected_test_output = np.array([9, -10])
    actual_test_output = layer.forward(input_vector)
    assert np.array_equal(actual_test_output, expected_test_output)
