""" Module for testing purposes """
from src.infra import DigitRecognizerRepository

from src.nn import Model, Layer, MeanSquaredError, ReLu

repository = DigitRecognizerRepository()
print("Loading the data...")
repository.load_data()
print(f"Train data: {repository.train_inputs.shape} | {repository.train_outputs.shape}")
print(f"Test data: {repository.test_inputs.shape} | {repository.test_outputs.shape}")

model = Model(
    repository,
    [
        Layer((1024, 784), ReLu()),
        Layer((1024, 1024), ReLu()),
        Layer((1024, 1024), ReLu()),
        Layer((10, 1024), ReLu()),
    ],
    MeanSquaredError(),
)

input_vector = repository.train_inputs[0]
print(f"Input: {input_vector}")
output_vector = model.run(input_vector)
print(f"Output: {output_vector}")
