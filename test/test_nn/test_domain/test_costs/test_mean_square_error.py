"""
    This module tests the funcionality of the Mean Squared Error cost function.
"""
import math
import numpy as np

from src.nn import MeanSquaredError


def test_mean_square_error_eval():
    """
    Cost function should return the correct value when evaluated.
    """
    cost = MeanSquaredError()
    predicted_output = np.array([1, 2, 3])
    sample_output = np.array([3, 2, 2])
    expected_test_output = 5 / 3
    actual_test_output = cost.eval(predicted_output, sample_output)
    assert math.isclose(actual_test_output, expected_test_output)


def test_mean_square_error_gradient():
    """
    Cost function should return the correct gradient.
    """
    cost = MeanSquaredError()
    predicted_output = np.array([1, 2, 3])
    sample_output = np.array([3, 2, 2])
    expected_test_output = (1 / 3) * np.array([-4, 0, 2])
    actual_test_output = cost.gradient(predicted_output, sample_output)
    assert np.array_equal(actual_test_output, expected_test_output)
