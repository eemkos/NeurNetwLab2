import numpy as np
from Augmentor.Operations import Operation
from PIL import Image
import random


class Shift(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, max_shift_horizontal, max_shift_vertical):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.max_shift_horizontal = max_shift_horizontal
        self.max_shift_vertical = max_shift_vertical

    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        image = np.asarray(image.convert("RGB"))

        shift_vertical = random.randint(0, self.max_shift_vertical)
        shift_horizontal = random.randint(0, self.max_shift_horizontal)

        image = self._shift_horizontal(image, shift_horizontal)
        image = self._shift_vertical(image, shift_vertical)

        return Image.fromarray(np.uint8(image)).convert('1')

    @staticmethod
    def _shift_horizontal(image, shift_horizontal):
        col_ones = np.ones((image.shape[0], 1, image.shape[2])) * 255
        if random.random() < 0.5:  # shift right
            for i in range(shift_horizontal):
                image = np.delete(image, image.shape[1] - 1, axis=1)
                image = np.concatenate([col_ones, image], axis=1)
        else:  # shift left
            for i in range(shift_horizontal):
                image = np.delete(image, 0, axis=1)
                image = np.concatenate([image, col_ones], axis=1)
        return image

    @staticmethod
    def _shift_vertical(image, shift_vertical):
        row_ones = np.ones((1, image.shape[1], image.shape[2])) * 255
        if random.random() < 0.5:  # shift down
            for i in range(shift_vertical):
                image = np.delete(image, image.shape[1] - 1, axis=0)
                image = np.concatenate([row_ones, image], axis=0)
        else:  # shift up
            for i in range(shift_vertical):
                image = np.delete(image, 0, axis=0)
                image = np.concatenate([image, row_ones], axis=0)
        return image