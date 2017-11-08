import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from Augmentor.Operations import Operation
from PIL import Image


class ElasticTransform(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, alpha, sigma, alpha_affine, random_state=None):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.random_state = random_state

    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        if self.random_state is None:
            random_state = np.random.RandomState(None)
        #image = np.asarray(image.convert('1')).astype(np.float32)
        #image = np.expand_dims(image, axis=2)
        #image = np.repeat(image, 3, axis=2)
        image = np.asarray(image.convert('RGB')).astype(np.float32)
        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


        # Return the image so that it can further processed in the pipeline:
        return Image.fromarray(np.uint8(image)).convert('1')