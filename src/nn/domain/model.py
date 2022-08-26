import numpy as np


class Model:
    def __init__(self, repository, layers, cost):
        self.repository = repository
        self.layers = layers
        self.cost = cost

    def load(self):
        pass

    def run(self, input_batch):
        output_batch = np.transpose(input_batch.copy())
        for layer in self.layers:
            output_batch = np.add(np.dot(layer.weights, output_batch), layer.bias)
        return np.transpose(output_batch)

    def test(self):
        testing_results = np.argmax(self.run(self.repository.test_inputs), axis=1)
        correct = sum(
            int(x == y) for (x, y) in zip(self.repository.test_outputs, testing_results)
        )
        return correct / self.repository.test_outputs.shape[0]

    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()

    def __str__(self):
        string = (
            "-" * 65
            + "\n"
            + f"Inputs: {self.layers[0].n_inputs}\n"
            + f"Outputs: {self.layers[len(self.layers)-1].n_neurons}\n"
            + f"Training data: {np.shape(self.repository.train_inputs)}\n"
            + f"Testing data: {np.shape(self.repository.test_inputs)}\n"
            + f"Cost: {self.cost}\n"
            + "Layers:"
        )
        for layer in self.layers:
            string += "\t" + f"{(layer.n_neurons, layer.n_inputs)} {layer.activation}\n"
        string += "-" * 65
        return string
