import random as rnd
from src.utils.math_utils import add, mult, dot, repeat_row
import numpy as np


def shuffle_data(x_train, y_train):
    indices = list(range(len(y_train)))
    rnd.shuffle(indices)
    return x_train[indices, :], y_train[indices, :]


def update_weights(old_weights, learn_coef, errors, inputs):
    return add(old_weights, mult(mult(inputs,
                                      repeat_row(errors, len(inputs))),
                                 learn_coef))


def accuracy(y, output):
    y_lbl = np.argmax(y,axis=1)
    o_lbl = np.argmax(output, axis=1)

    return 100.*np.sum(y_lbl == o_lbl) /len(y_lbl)
