from abc import ABC, abstractmethod


class Repository(ABC):
    def __init__(self):
        self.train_inputs = None
        self.train_outputs = None
        self.test_inputs = None
        self.test_outputs = None

    @abstractmethod
    def load_data(self):
        pass
