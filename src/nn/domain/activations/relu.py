from ..activation import Activation


class ReLu(Activation):
    def __init__(self):
        super().__init__("ReLu")

    def eval_element(self, input):
        return max(0, input)

    def derivative_element(self, input):
        if input == 0:
            raise ValueError
        return 0 if input < 0 else 1
