import numpy as np


def bengio_tanh(nb_inputs, nb_neurons):
    r = _r(nb_inputs, nb_neurons)
    return np.random.uniform(low=-r, high=r, size=(nb_inputs, nb_neurons))


def bengio_sigm(nb_inputs, nb_neurons):
    r = 4.0 * _r(nb_inputs, nb_neurons)
    return np.random.uniform(low=-r, high=r, size=(nb_inputs, nb_neurons))


def _r(nb_inputs, nb_neurons):
    return np.sqrt(6.0/(nb_inputs + nb_neurons))


