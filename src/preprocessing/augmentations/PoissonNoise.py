import numpy as np
from Augmentor.Operations import Operation
from PIL import Image


class PoissonNoise(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, min_peak=0.95, max_peak=1.0):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.min_peak = min_peak
        self.max_peak = max_peak

    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        image = np.asarray(image) / 255.
        peak = np.random.uniform(self.min_peak, self.max_peak)
        noisy = np.random.poisson(image * peak * 255.) / peak
        noisy = np.clip(noisy, 0.0, 255.0)

        return Image.fromarray(np.uint8(noisy)).convert('L')

#def poisson_noise(x):
#    """
#    According to github I lost link to
#    """
#    peak = np.random.uniform(0.95, 1.0)
#    noisy = np.random.poisson(x * 255.0 * peak) / peak / 255.0
#    noisy = np.clip(noisy, 0.0, 1.0)
#    return noisy