import numpy as np

from src.utils.math_utils import exp


class Elu:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.name = 'Elu'

    def activate(self, net):
        result = np.asarray(net)
        for i in range(len(net)):
            for j in range(net.shape[1]):
                if net[i,j] <= 0:
                    result[i,j] = self.alpha * (exp(net[i,j]) - 1)
        return result

    def gradient(self, net):
        result = np.ones(net.shape)
        for i in range(len(net)):
            for j in range(net.shape[1]):
                if net[i,j] <= 0:
                    result[i,j] = self.alpha * (exp(net[i,j]) - 1) + self.alpha
        return result