
from src.model import Model, Layer
from src.optimization.optimizers import StochasticGradientDescent
from src.optimization.losses import Crossentropy
from src.model.activations import Softplus, Sigmoid
from src.optimization.momentum import NoMomentum
from src.optimization.callbacks import AccuracyStoppingCallback
from src.data_access import val_data, train_data
import numpy as np
from src.utils.training_utils import shuffle_data
import pandas as pd

reps = 10
req_acc=85


hid_sizes = [10, 26, 35, 40, 50, 100, 200]

size_epochs = {'size':[], 'epochs_mean': [], 'epochs_std': []}


def prepare_data(data, data_part=1.0):
    x, y_lbl = data
    y = np.zeros((len(y_lbl), 10))
    y[np.arange(len(y_lbl)), y_lbl.flatten().astype(int)] = 1

    x, y = shuffle_data(x, y)
    ln = int(data_part*len(y))
    x = x[:ln]
    y = y[:ln]
    return x, y


j = 0
for size in hid_sizes:
    j+=1
    epochses = []
    for i in range(reps):
        lrs = [Layer(nb_neurons=70),
               Layer(nb_neurons=size, activation_function=Softplus(), use_bias=True),
               Layer(nb_neurons=10, activation_function=Sigmoid(), use_bias=True)]

        mdl = Model(lrs)

        sgd = StochasticGradientDescent(model=mdl, train_coeff=0.1,
                                        momentum=NoMomentum(),
                                        batch_size=50, loss=Crossentropy(Sigmoid()),
                                        max_epochs=100,
                                        callbacks=[AccuracyStoppingCallback(req_acc)]
                                        )

        x_t, y_t = prepare_data(train_data(True, True))
        x_v, y_v = prepare_data(val_data())
        sgd.train(x_t,y_t,x_v,y_v)

        epochses.append(sgd.epoch)
        print('%d/%d sizes ---- %d repetition' % (j, len(hid_sizes), i))

    size_epochs['size'].append(size)
    size_epochs['epochs_mean'] = np.mean(epochses)
    size_epochs['epochs_std'] = np.std(epochses)


df = pd.DataFrame(size_epochs)
df.to_csv('./logs/nb_hidden_neurons_tests/results.csv')


