import numpy as np
import os
from PIL import Image

IMAGE_ROW_SHAPE = (1, 70)
EXPECTED_IMAGE_NUMPY_SHAPE = (10, 7)
EXPECTED_IMAGE_SHAPE = (7, 10)

TRAIN_DIR = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2data/train/'
AUGM_DIR = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2data/augmented/'
VAL_DIR = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2data/val/'


def repair_wrong_format_images(path=TRAIN_DIR):
    newimages = []
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            img = Image.open(path + filename)
            if img.size != EXPECTED_IMAGE_SHAPE:
                img = img.resize(EXPECTED_IMAGE_SHAPE, Image.ANTIALIAS)
                newimages.append((filename, img))

    for filename, img in newimages:
        img.save(path + filename)
