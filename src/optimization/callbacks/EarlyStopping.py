from .BaseCallback import BaseCallback
import numpy as np


class EarlyStopping(BaseCallback):
    def __init__(self, patience=1, save_best_only=True):
        super().__init__()
        self.patience = patience
        self.save_best_only = save_best_only

        self.val_losses_history = []
        self.weights_history = []

        self.early_stopped = False

    def on_training_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_epoch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_batch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_batch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_epoch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        self.val_losses_history.append(optimizer.current_val_loss)
        self.weights_history.append(self.copy_weights(model))

        if len(self.val_losses_history) > self.patience:
            if self.val_losses_history[-1-self.patience] < self.val_losses_history[-1]:
                optimizer.should_stop = True
                self.early_stopped = True

    def on_training_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        if self.early_stopped and self.save_best_only:
            print('Restoring best weights')
            self.restore_best_weights(model)

    def restore_best_weights(self, model):
        minind = 0
        minloss = self.val_losses_history[0]

        for i in range(len(self.val_losses_history)):
            if self.val_losses_history[i] <= minloss:
                minloss = self.val_losses_history[i]
                minind = i

        print('Restored weights from epoch %d'%(minind))
        best_weights = self.weights_history[minind]

        for l in range(len(model.layers)):
            model.layers[l].weights = best_weights[l]

    @staticmethod
    def copy_weights(model):
        weights = []
        for layer in model.layers:
            weights.append(np.copy(layer.weights))
        return weights

