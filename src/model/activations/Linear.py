from src.utils.math_utils import mult


class Linear:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.name = 'Linear'

    def activate(self, net):
        return mult(net, self.alpha)

    def gradient(self, net):
        return self.alpha