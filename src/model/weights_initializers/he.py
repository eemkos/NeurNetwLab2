import numpy as np


def he_uniform(nb_inputs, nb_neurons):
    r = np.sqrt(6.0/nb_inputs)
    return np.random.uniform(low=-r, high=r, size=(nb_inputs, nb_neurons))


def he_normal(nb_inputs, nb_neurons):
    r = np.sqrt(1.0/nb_inputs)
    return np.random.normal(loc=0, scale=r, size=(nb_inputs, nb_neurons))




