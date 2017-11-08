from src.utils.math_utils import add, exp, mult, subtract, div


class Tanh:
    def __init__(self, beta=1):
        self.beta = beta
        self.name = 'Tanh'

    def activate(self, net):
        divisor = add(exp(mult(-self.beta, net)), 1)
        return subtract(div(2, divisor), 1)

    def gradient(self, net):
        tnh = self.activate(net)
        return mult(self.beta, subtract(1, mult(tnh, tnh)))