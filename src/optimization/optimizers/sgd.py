from src.optimization.callbacks.EpochLossCalculationCallback import EpochLossCalculationCallback
from src.optimization.momentum.NoMomentum import NoMomentum
from src.utils.math_utils import mult, dot, div, transpose, subtract, add
from src.utils.training_utils import accuracy
from src.utils.training_utils import shuffle_data


class StochasticGradientDescent:

    def __init__(self, train_coeff, model, loss,
                 starting_epoch=0, max_epochs=10,
                 batch_size=32,
                 callbacks=[],
                 momentum=NoMomentum(),
                 regularisation_rate=0.0):

        self.train_coeff = train_coeff
        self.model = model
        self.loss = loss
        self.model.loss = loss

        self.batch = 0
        self.nb_batches = 0
        self.batch_size = batch_size

        self.epoch = starting_epoch
        self.nb_epochs = max_epochs

        self.momentum = momentum
        self.regularisation_rate=regularisation_rate
        self.should_stop = False

        self.current_val_loss = 0
        self.current_val_acc = 0
        self.current_train_loss = 0
        self.current_train_acc = 0

        self.callbacks = [EpochLossCalculationCallback()] + callbacks

    def train(self, x_train, y_train, x_val, y_val):

        self.nb_batches = int(len(x_train) / self.batch_size)

        for callback in self.callbacks:
            callback.on_training_starting(self.model, self, x_train, y_train, x_val, y_val)

        while self.epoch < self.nb_epochs and not self.should_stop:
            x_train, y_train = shuffle_data(x_train, y_train)
            self.epoch += 1
            self.batch = 0
            batch_start_index = 0

            for callback in self.callbacks:
                callback.on_epoch_starting(self.model, self, x_train, y_train, x_val, y_val)

            while batch_start_index < len(y_train):
                for callback in self.callbacks:
                    callback.on_batch_starting(self.model, self, x_train, y_train, x_val, y_val)

                x_batch = x_train[batch_start_index:batch_start_index+self.batch_size]
                y_batch = y_train[batch_start_index:batch_start_index+self.batch_size]
                batch_start_index += self.batch_size

                o = self.model.feedforward(x_batch)
                _ = self.model.backpropagate(o, y_batch, self.loss)

                self.update_weights(len(x_batch), self.momentum)

                output = self.model.feedforward(x_val)
                self.current_val_loss = self.model.loss.loss(y_val, output)
                self.current_val_acc = accuracy(y_val, output)

                for callback in self.callbacks:
                    callback.on_batch_finished(self.model, self, x_train, y_train, x_val, y_val)
                self.batch += 1

            for callback in self.callbacks:
                callback.on_epoch_finished(self.model, self, x_train, y_train, x_val, y_val)


        for callback in self.callbacks:
            callback.on_training_finished(self.model, self, x_train, y_train, x_val, y_val)

    def update_weights(self, batch_size, momentum):
        r = self.train_coeff * self.regularisation_rate

        for layer in self.model.layers[1:]:
            modif = transpose(div(mult(self.train_coeff,
                                       dot(transpose(layer.last_errors),
                                           layer.last_inputs)),
                                  batch_size))
            mmntm = momentum.momentum(layer)
            regul = r * layer.weights

            layer.subtr_from_weights(subtract(modif, mmntm))
            layer.subtr_from_weights(regul)

