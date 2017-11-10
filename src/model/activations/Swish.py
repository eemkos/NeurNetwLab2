from src.utils.math_utils import add, mult, subtract, exp, div
import numpy as np

class Swish:
    def __init__(self, beta=1.):
        self.beta=beta
        self.name = 'Swish'

    def activate(self, net):

        return mult(net, self._sigm(net))

    def gradient(self, net):
        #print(np.max(net), np.min(net))
        beta_swsh = mult(self.beta, self.activate(net))
        sigm_beta = self._sigm(mult(self.beta, net))

        return add(beta_swsh, mult(sigm_beta, subtract(1, beta_swsh)))

    def _sigm(self, net):
        ex = exp(mult(-self.beta, net))
        return div(1, add(ex, 1))