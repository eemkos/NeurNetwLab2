import pickle as pkl


class Model:

    def __init__(self, layers):
        self.layers = layers

        for i1, i2 in zip(range(0,len(layers)-1),range(1,len(layers))):
            self.layers[i1].connect_consequent_layer(self.layers[i2])
            self.layers[i2].connect_preceding_layer(self.layers[i1])

    def train(self):
        pass

    def feedforward(self, x):
        curr_activ = x
        for layer in self.layers:
            #print(curr_activ.shape)
            curr_activ = layer.feedforward(curr_activ)

        return curr_activ

    def backpropagate(self, output, expected, loss):
        currerr = loss.gradient_over_net(expected, output, self.layers[-1].last_nets)
        for layer in self.layers[1:][::-1]:
            currerr = layer.backprop_error(currerr)

        return currerr

    def save_weights(self, filepath):
        weights = [l.weights for l in self.layers]

        with open(filepath, 'wb') as f:
            pkl.dump(weights, f)

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            weights = pkl.load(f)

        for i in range(len(weights)):
            self.layers[i].weights = weights[i]

    def save_model(self, filepath, with_weights=True):
        if with_weights:
            with open(filepath, 'wb') as f:
                pkl.dump(self, f)
        else:
            pass
            #weights = [l.weights for l in self.layers]
            #for l in self.layers:
            #    l.weights=None

    @staticmethod
    def load_model(filepath, with_weights=True):
        with open(filepath, 'rb') as f:
            return pkl.load(f)




