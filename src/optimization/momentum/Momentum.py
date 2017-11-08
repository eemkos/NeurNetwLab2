from src.utils.math_utils import subtract, mult


class Momentum:

    def __init__(self, momentum_coeff):
        self.coeff = momentum_coeff

    def momentum(self, layer):
        delta = subtract(layer.weights, layer.last_weights)
        return mult(self.coeff, delta)