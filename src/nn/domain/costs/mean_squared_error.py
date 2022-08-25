import numpy as np

from ..cost import Cost


class MeanSquaredError(Cost):
    """
    The Mean Square Error cost function estimates the cost by summing
    the squared differences of the predicted value and the sample value
    """

    def __init__(self):
        super().__init__("Mean Squared Error")

    def eval(self, predicted_output: np.ndarray, sample_output: np.ndarray):
        """Evaluation of the Mean Square Error cost function"""
        return (1 / len(predicted_output)) * np.sum(
            (predicted_output - sample_output) ** 2
        )

    def gradient(self, predicted_output: np.ndarray, sample_output: np.ndarray):
        """Derivative of the Mean Square Error cost function"""
        return (2 / len(predicted_output)) * np.subtract(
            predicted_output, sample_output
        )
