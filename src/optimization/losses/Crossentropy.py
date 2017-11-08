from src.utils.math_utils import calc_sum, mult, ln, subtract


class Crossentropy:

    def __init__(self, activation=None):
        self.activation = activation

    def loss(self, y, out):
        return -1. * calc_sum(mult(y, ln(out))) / (len(y))

    def gradient_over_net(self, y, out, net):
        if self.activation.name in ['Softmax', 'Sigmoid']:
            return subtract(out, y)
        else:
            pass

    def gradient_over_activation(self):
        pass