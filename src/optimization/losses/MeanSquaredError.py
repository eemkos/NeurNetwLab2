from src.utils.math_utils import subtract, mult, calc_sum


class MeanSquaredError:

    def __init(self, activation=None):
        self.activation = activation

    def loss(self, y, out):
        diff = subtract(y, out)
        sq = mult(diff, diff)
        return 1. * calc_sum(sq) / len(y)

    def gradient_over_net(self, y, out, net):
        return mult(subtract(y, out), self.activation.gradient(net))

    def gradient_over_activation(self, y, out):
        return subtract(y, out)