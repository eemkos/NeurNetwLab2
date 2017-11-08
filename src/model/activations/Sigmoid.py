from src.utils.math_utils import add, mult, subtract, exp, div


class Sigmoid:
    def __init__(self, beta=1):
        self.beta=beta
        self.name = 'Sigmoid'

    def activate(self, net):
        ex = exp(mult(-self.beta, net))
        return div(1, add(ex, 1))

    def gradient(self, net):
        sgm = self.activate(net)
        return mult(self.beta, mult(sgm, subtract(1, sgm)))