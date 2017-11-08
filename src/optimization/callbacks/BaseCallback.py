

class BaseCallback:

    def __init__(self):
        pass

    def on_training_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_epoch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_batch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_batch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_epoch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_training_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

