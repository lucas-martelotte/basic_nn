"""
This module implements a repository to load data for a digit recognizer AI.
The data is taken from this site: https://www.kaggle.com/competitions/digit-recognizer
"""
import pandas as pd
import numpy as np

from ..nn import Repository


class DigitRecognizerRepository(Repository):
    """
    The repository responsible for loading the digit recognizer train and test data.
    """

    def load_data(self):
        train_data = pd.read_csv("./data/digit_recognizer/train.csv")
        train_data = np.array(train_data)
        test_inputs = pd.read_csv("./data/digit_recognizer/test.csv")
        test_inputs = np.array(test_inputs)
        test_outputs = pd.read_csv("./data/digit_recognizer/sample_submission.csv")
        test_outputs = np.array(test_outputs)
        self.train_outputs = train_data[:, 0]
        self.train_inputs = train_data[:, 1:]
        self.test_outputs = test_outputs[:, 1]
        self.test_inputs = test_inputs
