from src.utils.math_utils import exp, ln, add, neg, div


class Softplus:
    def __init__(self):
        self.name = 'Softplus'

    def activate(self, net):
        ex = exp(net)
        return ln(add(ex, 1))

    def gradient(self, net):
        ex = exp(neg(net))
        return div(1, add(ex, 1))