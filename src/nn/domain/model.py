import numpy as np


class Model:
    def __init__(self, sample_inputs, sample_outputs, layers, cost):
        self.sample_inputs = np.array(sample_inputs)
        self.sample_outputs = np.array(sample_outputs)
        self.layers = layers
        self.cost = cost

    def train(self):
        pass

    def load(self):
        pass

    def run(self, input):
        output = input.copy()
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def __str__(self):
        string = (
            "-" * 65
            + "\n"
            + f"Inputs: {self.layers[0].n_inputs}\n"
            + f"Outputs: {self.layers[len(self.layers)-1].n_neurons}\n"
            + f"Training data: {len(self.sample_inputs)}\n"
            + f"Cost: {self.cost}\n"
            + f"Layers:"
        )
        for layer in self.layers:
            string += "\t" + f"{(layer.n_neurons, layer.n_inputs)} {layer.activation}\n"
        string += "-" * 65
        return string
