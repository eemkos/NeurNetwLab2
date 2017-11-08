from .BaseCallback import BaseCallback
from src.utils.training_utils import accuracy


class EpochLossCalculationCallback(BaseCallback):

    def __init__(self):
        super().__init__()

    def on_training_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        output_train = model.feedforward(x_train)
        optimizer.current_train_loss = model.loss.loss(y_train, output_train)
        optimizer.current_train_acc = accuracy(y_train, output_train)

        output_val = model.feedforward(x_val)
        optimizer.current_val_loss = model.loss.loss(y_val, output_val)
        optimizer.current_val_acc = accuracy(y_val, output_val)

    def on_epoch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_batch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_batch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_epoch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        output_train = model.feedforward(x_train)
        optimizer.current_train_loss = model.loss.loss(y_train, output_train)
        optimizer.current_train_acc = accuracy(y_train, output_train)

        output_val = model.feedforward(x_val)
        optimizer.current_val_loss = model.loss.loss(y_val, output_val)
        optimizer.current_val_acc = accuracy(y_val, output_val)

    def on_training_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass
