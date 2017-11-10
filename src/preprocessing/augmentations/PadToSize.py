import numpy as np
from Augmentor.Operations import Operation
from PIL import Image


class PadToSize(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, height, width):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.height = height
        self.width = width

    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        image = np.asarray(image.convert("RGB"))
        assert image.shape[0] <= self.height and image.shape[1] <= self.width

        col_ones = np.ones((image.shape[0], 1, image.shape[2])) * 255

        while image.shape[1] < self.width:
            #print(col_ones.shape, image.shape)
            image = np.concatenate([col_ones, image], axis=1)
            if image.shape[1] < self.width:
                image = np.concatenate([image, col_ones], axis=1)

        row_ones = np.ones((1, image.shape[1], image.shape[2])) * 255
        while image.shape[0] < self.height:
            image = np.concatenate([row_ones, image], axis=0)
            if image.shape[0] < self.height:
                image = np.concatenate([image, row_ones], axis=0)

        return Image.fromarray(np.uint8(image)).convert('L')
