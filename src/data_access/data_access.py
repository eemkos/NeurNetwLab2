import numpy as np
import os
from PIL import Image

IMAGE_ROW_SHAPE = (1, 70)
EXPECTED_IMAGE_NUMPY_SHAPE = (10, 7)
EXPECTED_IMAGE_SHAPE = (7, 10)

TRAIN_DIR = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2/data/train/'
AUGM_DIR = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2/data/augmented/'
MAN_DIR = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2/data/my_generated/'
VAL_DIR = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2/data/val/'


def train_data(use_augmented=True, use_manually_generated=True):
    list_samples = []
    for label in range(10):
        list_samples += parse_images(TRAIN_DIR, label)

        if use_augmented:
            list_samples += parse_images(AUGM_DIR, label)

        if use_manually_generated:
            list_samples += parse_images(MAN_DIR, label)

    return list_to_arrays(list_samples)


def val_data():
    list_samples = []
    for label in range(10):
        list_samples += parse_images(VAL_DIR, label)

    return list_to_arrays(list_samples)


def parse_images(path=TRAIN_DIR, label=0):
    path = path + str(label) + '/'
    return[(_imgpath_to_np_array(path + filename), label) for filename in os.listdir(path) if filename.endswith('.png')]


def list_to_arrays(data):
    images = np.empty((len(data), IMAGE_ROW_SHAPE[1]))
    labels = np.empty((len(data)))
    for i in range(len(data)):
        images[i] = data[i][0]
        labels[i] = data[i][1]

    return images, labels


def _imgpath_to_np_array(imgpath):
    return np.asarray(Image.open(imgpath).convert('1')).astype(int).reshape(IMAGE_ROW_SHAPE)


def show_imgs(limit=10, path=TRAIN_DIR):
    for filename in os.listdir(path):
        limit -= 1
        if limit < 0: break
        gg = np.asarray(Image.open(path + filename).convert('1')).astype(int)
        show_img(gg)


def show_img(gg):
    for i in range(gg.shape[0]):
        line = ''
        for j in range(gg.shape[1]):
            if gg[i, j] < 0.5:
                line += ' # '
            else:
                line += '   '
        print(line)
    print('\n')