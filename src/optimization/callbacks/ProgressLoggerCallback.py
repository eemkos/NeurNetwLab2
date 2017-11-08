from .BaseCallback import BaseCallback
import progressbar as pb
import time


class ProgressLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.pbar = None
        self.widgets = None

    def on_training_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_epoch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        self.widgets = [pb.FormatLabel(''), ' ',
                        pb.Percentage(), ' ',
                        pb.Bar('#'), ' ',
                        pb.RotatingMarker(), ' ',
                        pb.FormatLabel('')
                        ]
        self.pbar = pb.ProgressBar(maxval=optimizer.nb_batches, widgets=self.widgets,
                                   redirect_stderr=True, redirect_stdout=True,
                                   custom_len=lambda x: 3)
        self.pbar.start()

    def on_batch_starting(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    def on_batch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        self.widgets[0] = self.batch_progress_label(optimizer)
        self.widgets[-1] = self.losses_accuracies_label(optimizer)
        self.pbar.update(optimizer.batch)

    def on_epoch_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        self.pbar.finish()
        pb.streams.flush()
        pb.streams.wrap_stdout()
        redirect_stdout = True
        time.sleep(0.1)
        print()
        print('Epoch %d / %d passed' % (optimizer.epoch, optimizer.nb_epochs),
              '--- train_loss:', optimizer.current_train_loss,
              ' --- train_acc:', optimizer.current_train_acc,
              '--- val_loss:', optimizer.current_val_loss,
              ' --- val_acc:', optimizer.current_val_acc
              )

    def on_training_finished(self, model, optimizer, x_train, y_train, x_val, y_val):
        pass

    @staticmethod
    def batch_progress_label(optimizer):

        return pb.FormatLabel(('  Batch %d / %d passed' % (optimizer.batch, optimizer.nb_batches)).ljust(25))

    @staticmethod
    def losses_accuracies_label(optimizer):
        return pb.FormatLabel(
            'train_loss: %f, train_acc: %f, val_loss: %f, val_acc: %f' % (optimizer.current_train_loss,
                                                                          optimizer.current_train_acc,
                                                                          optimizer.current_val_loss,
                                                                          optimizer.current_val_acc))
