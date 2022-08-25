"""
    This module tests the funcionality of the ReLu activation function.
"""
import pytest

from src.nn import ReLu


def test_relu_eval_negative_element():
    """
    Relu should clip negative numbers to zero.
    """
    activation = ReLu()
    assert activation.eval_element(-1) == 0
    assert activation.eval_element(-421) == 0
    assert activation.eval_element(-24) == 0


def test_relu_eval_non_negative_element():
    """
    Relu should do nothing with non negative elements.
    """
    activation = ReLu()
    assert activation.eval_element(520) == 520
    assert activation.eval_element(0) == 0
    assert activation.eval_element(5) == 5


def test_relu_derivative_negative_element():
    """
    ReLu's derivative should clip negative numbers to zero.
    """
    activation = ReLu()
    assert activation.derivative_element(-2) == 0
    assert activation.derivative_element(-321) == 0
    assert activation.derivative_element(-51) == 0


def test_relu_derivative_positive_element():
    """
    ReLu's derivative should clip positive numbers to one.
    """
    activation = ReLu()
    assert activation.derivative_element(3) == 1
    assert activation.derivative_element(162) == 1
    assert activation.derivative_element(22) == 1


def test_relu_derivative_zero():
    """
    ReLu has no derivative at zero. It should raise an error.
    """
    activation = ReLu()
    with pytest.raises(ValueError):
        activation.derivative_element(0)
