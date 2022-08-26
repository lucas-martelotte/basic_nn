"""
    This module tests the funcionality of the Sigmoid activation function.
"""
import math

from src.nn import Sigmoid


def test_sigmoid_eval_element():
    """
    Sigmoid should evaluate correctly.
    """
    activation = Sigmoid()
    assert math.isclose(activation.eval_element(0), 1 / 2)
    assert math.isclose(activation.eval_element(-1), 1 / (1 + math.e))
    assert math.isclose(activation.eval_element(3), 1 / (1 + math.pow(math.e, -3)))


def test_sigmoid_derivative_element():
    """
    Sigmoid should calculate the derivative correctly.
    """
    activation = Sigmoid()
    assert math.isclose(activation.derivative_element(0), 1 / 4)
    function_value = activation.eval_element(-2)
    assert math.isclose(
        activation.derivative_element(-2), function_value * (1 - function_value)
    )
    function_value = activation.eval_element(3)
    assert math.isclose(
        activation.derivative_element(3), function_value * (1 - function_value)
    )
