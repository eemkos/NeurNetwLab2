import numpy as np

from src.data_access import val_data, train_data
from src.executor.config import config
from src.executor.config_parser import parse_loss, parse_optimizer, parse_model
from src.model import Model
from src.utils.digit_painter import PaintWindow
from src.utils.training_utils import shuffle_data, accuracy

repetition = 0


def run():
    global repetition
    if config['repetitions'] is None:
        rep = 1
    else:
        rep = config['repetitions']

    for i in range(rep):
        repetition = i
        for mode in config['execution_plan']:
            if mode == 'training_mode':
                train()
            elif mode == 'testing_mode':
                test()
            elif mode == 'user_mode':
                user_mode()


def test():
    tc = config['testing_mode']
    model = Model.load_model(tc['model_filepath'])

    x, y= prepare_data(val_data())

    print("Val Data Len:", len(x))

    out = model.feedforward(x)
    acc = accuracy(y, out)

    print("Accuracy: %f" % acc, '%')


def user_mode():
    uc = config['user_mode']
    model = Model.load_model(uc['model_filepath'])

    for i in range(uc['nb_trials']):
        ptw = PaintWindow()
        arr = ptw().reshape((1, 70))
        res = model.feedforward(arr)
        print(np.argmax(res[0, :]))


def train():
    trc = config['training_mode']
    mc = config['model_config']

    if trc['train_from_epoch'] == 0:
        start_new_training(trc, mc)
    else:
        continue_training(trc)


def start_new_training(tr_conf, m_conf):
    model = parse_model(m_conf)
    loss = parse_loss(model, tr_conf)
    optimizer = parse_optimizer(model, loss, tr_conf['optimizer'], repetition)

    x_tr, y_tr = prepare_data(train_data(use_augmented=tr_conf['use_augmented_data'],
                                            use_manually_generated=tr_conf['use_self_generated_data']),
                                 data_part=tr_conf['part_of_data'])
    x_val, y_val = prepare_data(val_data())
    print('Train data len: ', len(x_tr))
    print('Val data len: ', len(x_val))

    optimizer.train(x_tr, y_tr, x_val, y_val)
    model.save_model(tr_conf['save_model_filepath'], True)


def continue_training(tr_conf):
    model = Model.load_model(tr_conf['load_model_filepath'])
    loss = parse_loss(model, tr_conf)
    optimizer = parse_optimizer(model, loss, tr_conf['optimizer'], repetition)

    x_tr, y_tr = prepare_data(train_data(use_augmented=tr_conf['use_augmented_data'],
                                            use_manually_generated=tr_conf['use_self_generated_data']))
    x_val, y_val = prepare_data(val_data())

    optimizer.train(x_tr, y_tr, x_val, y_val)
    model.save_model(tr_conf['save_model_filepath'], True)


def prepare_data(data, data_part=1.0):
    x, y_lbl = data
    y = np.zeros((len(y_lbl), 10))
    y[np.arange(len(y_lbl)), y_lbl.flatten().astype(int)] = 1

    x, y = shuffle_data(x, y)
    ln = int(data_part*len(y))
    x = x[:ln]
    y = y[:ln]
    return x, y
