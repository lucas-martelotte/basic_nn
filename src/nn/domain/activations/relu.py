from ..activation import Activation


class ReLu(Activation):
    def __init__(self):
        super().__init__("ReLu")

    def eval_element(self, value):
        return max(0, value)

    def derivative_element(self, value):
        if value == 0:
            raise ValueError
        return 0 if value < 0 else 1
