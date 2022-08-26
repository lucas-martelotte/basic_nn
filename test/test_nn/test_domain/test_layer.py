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
    assert np.shape(layer.weights) == (4, 3)


def test_layer_reset_weights():
    """
    Layer's weights should be different after calling reset_weights.
    """
    layer = Layer((4, 3), ReLu())
    before_weights = layer.weights.copy()
    layer.reset_weights()
    assert not np.array_equal(layer.weights, before_weights)
