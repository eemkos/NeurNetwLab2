import numpy as np

from src.utils.math_utils import exp, div


class Softmax:
    def __init__(self):
        self.name = 'Softmax'

    def activate(self, net):
        ex = exp(net)
        sums = np.expand_dims(np.sum(ex, axis=1), ex.shape[1])
        return div(ex, np.repeat(sums, ex.shape[1], axis=1))

    def gradient(self, net):
        pass