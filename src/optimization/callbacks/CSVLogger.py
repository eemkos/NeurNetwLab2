from .BaseCallback import BaseCallback
import pandas as pd
import time


class CSVLogger(BaseCallback):

    def __init__(self, output_path, log_time=True):
        super().__init__()
        self.output_path = output_path

        self.epochs_log = {'Epoch': [],
                           'TrainLoss': [], 'TrainAccuracy': [],
                           'ValLoss': [], 'ValAccuracy': []}

        self.log_time = log_time
        if log_time:
            self.epochs_log['Timestamp'] = []
            self.epochs_log['EpochExecutionTime'] = []

        self.time_start = None
        self.current_epoch_end_time = None

    def on_training_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        self.epochs_log['Epoch'].append(0)
        self.epochs_log['TrainLoss'].append(optimizer.current_train_loss)
        self.epochs_log['TrainAccuracy'].append(optimizer.current_train_acc)
        self.epochs_log['ValLoss'].append(optimizer.current_val_loss)
        self.epochs_log['ValAccuracy'].append(optimizer.current_val_acc)

        if self.log_time:
            self.epochs_log['Timestamp'].append(0)
            self.epochs_log['EpochExecutionTime'].append(0)
            self.time_start = time.time()
            self.current_epoch_end_time = self.time_start

    def on_epoch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_batch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_batch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_epoch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        self.epochs_log['Epoch'].append(optimizer.epoch)
        self.epochs_log['TrainLoss'].append(optimizer.current_train_loss)
        self.epochs_log['TrainAccuracy'].append(optimizer.current_train_acc)
        self.epochs_log['ValLoss'].append(optimizer.current_val_loss)
        self.epochs_log['ValAccuracy'].append(optimizer.current_val_acc)

        if self.log_time:
            tm = time.time()
            self.epochs_log['Timestamp'].append(tm - self.time_start)
            self.epochs_log['EpochExecutionTime'].append(tm - self.current_epoch_end_time)
            self.current_epoch_end_time = tm

    def on_training_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        df = pd.DataFrame(self.epochs_log)
        df.to_csv(self.output_path, index=False)
