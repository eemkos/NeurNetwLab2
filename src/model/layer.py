import numpy as np
from src.utils.math_utils import dot, mult, subtract, transpose


class Layer:
    def __init__(self, nb_neurons, activation_function=None,
                 weights_initializer=None, use_bias=True):

        self.activation = activation_function
        self.use_bias = use_bias

        self.feedforward = self._first_layer_feedforward
        self.backprop_error = self._last_layer_backprop

        self.nb_neurons = nb_neurons
        self.nb_inputs = nb_neurons + 1 if self.use_bias else 0

        self.weights = np.ones((self.nb_inputs, self.nb_neurons))
        self.weights_initializer = self._default_weights_init if weights_initializer is None else weights_initializer

        self.preceding_layer = None
        self.consequent_layer = None

        self.last_weights = self.weights
        self.last_inputs = np.zeros((1, self.nb_inputs))
        self.last_nets = np.zeros((1, self.nb_neurons))
        self.last_activations = np.zeros((1, self.nb_neurons))
        self.last_errors = np.zeros((1, nb_neurons))

    def connect_preceding_layer(self, preceding_layer):
        self.nb_inputs = preceding_layer.nb_neurons + 1 if self.use_bias else 0
        self.weights = self.weights_initializer(self.nb_inputs, self.nb_neurons)
        self.last_weights = self.weights
        self.preceding_layer = preceding_layer
        self.feedforward = self._mid_layer_feedforward

    def connect_consequent_layer(self, consequent_layer):
        self.consequent_layer = consequent_layer
        self.backprop_error = self._mid_layer_backprop

    def subtr_from_weights(self, delta):
        self.last_weights = self.weights
        self.weights = subtract(self.weights, delta)

    def weights_without_bias(self):
        if self.use_bias:
            return np.delete(self.weights, 0, axis=0)
        else:
            return self.weights


    def _first_layer_feedforward(self, input):

        self.last_inputs = input
        self.last_nets = input
        self.last_activations = input
        return self.last_activations

    def _mid_layer_feedforward(self, input):
        if self.use_bias:
            input = np.c_[np.ones((input.shape[0], 1)), input]
        self.last_inputs = input
        self.last_nets = dot(input, self.weights)
        self.last_activations = self.activation.activate(self.last_nets)
        return self.last_activations

#    def _mid_layer_backprop(self, cons_layer_errors):
#        print(cons_layer_errors.shape, self.consequent_layer.weights.shape)
#        self.last_errors = mult(self.activation.gradient(self.last_nets),
#                                dot(cons_layer_errors,
#                                    transpose(self.consequent_layer.weights)))
#        return self.last_errors

    def _mid_layer_backprop(self, cons_layer_errors):
        a_gr = self.activation.gradient(self.last_nets)
        dt_e = dot(cons_layer_errors, transpose(self.consequent_layer.weights_without_bias()))
        self.last_errors = mult(a_gr, dt_e)

        return self.last_errors

    def _last_layer_backprop(self, loss_grad_activ_grad_error):
        self.last_errors = loss_grad_activ_grad_error
        return self.last_errors

    @staticmethod
    def _default_weights_init(nb_inp, nb_neur):
        return (np.random.random((nb_inp, nb_neur)) * .2) - .1