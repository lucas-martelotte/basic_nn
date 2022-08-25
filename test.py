""" Module for testing purposes """
import numpy as np

from src.nn import Model, Layer, MeanSquaredError, ReLu

model = Model(
    np.transpose([[1, 2, 3, 4], [2, 3, 4, 5], [4, 3, 3, 2]]),
    np.transpose([[1, 0, 0], [0, 0, 1], [1, 0, 1]]),
    [Layer((2, 4), ReLu()), Layer((2, 2), ReLu()), Layer((3, 2), ReLu())],
    MeanSquaredError(),
)

input = [1, 2, 1, 1]

print(model)
print(f"Input: {input}")
print(f"Output: {model.run(input)}")
