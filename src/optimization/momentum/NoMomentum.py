import numpy as np

class NoMomentum:

    def __init__(self, momentum_coeff=0):
        pass

    def momentum(self, layer):
        return np.zeros(layer.weights.shape)