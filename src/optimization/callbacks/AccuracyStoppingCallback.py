from .BaseCallback import BaseCallback


class AccuracyStoppingCallback(BaseCallback):

    def __init__(self, required_accuracy=80):
        super().__init__()
        self.required_accuracy = required_accuracy

    def on_epoch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        if optimizer.current_val_acc > self.required_accuracy:
            optimizer.should_stop = True
