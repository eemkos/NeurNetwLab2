import numpy as np


def lecture_initialization(nb_inputs, nb_neurons):
    mx = 1.0/np.sqrt(nb_inputs)
    mn = -mx

    return np.random.uniform(low=mn, high=mx, size=(nb_inputs, nb_neurons))