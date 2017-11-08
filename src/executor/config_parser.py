from src.executor.config import config
from src.model import Layer, Model
from src.model.activations import *
from src.model.weights_initializers import *
from src.optimization.callbacks import ProgressLoggerCallback, EarlyStopping, CSVLogger
from src.optimization.losses import Crossentropy, MeanSquaredError
from src.optimization.momentum import Momentum, NoMomentum
from src.optimization.optimizers import StochasticGradientDescent as SGD


def parse_loss(model, tr_conf):
    loss_name = tr_conf['loss']
    activation = model.layers[-1].activation

    if loss_name == 'Crossentropy':
        return Crossentropy(activation)
    elif loss_name == 'MeanSquaredError':
        return MeanSquaredError()
    else:
        return Crossentropy(activation)


def parse_optimizer(model, loss, opt_conf, repetition):
    callbacks = parse_callbacks(opt_conf['callbacks'], repetition)
    momentum = parse_momentum(opt_conf['momentum'])

    if opt_conf['name'] == 'SGD':
        return SGD(train_coeff=opt_conf['train_coefficient'],
                   model=model,
                   loss=loss,
                   max_epochs=opt_conf['max_epochs'],
                   batch_size=opt_conf['batch_size'],
                   momentum=momentum,
                   callbacks=callbacks)


def parse_callbacks(cbk_conf, repetition):
    cbks = []
    if cbk_conf['log_progress']['use']:
        cbks.append(ProgressLoggerCallback())

    es_conf = cbk_conf['early_stopping']
    if es_conf['use']:
        cbks.append(EarlyStopping(patience=es_conf['patience'],
                                  save_best_only=es_conf['best_only']))

    if cbk_conf['log_csv']['use']:
        if config['repetitions'] is None:
            cbks.append(CSVLogger(output_path=cbk_conf['log_csv']['log_path'],
                                  log_time=cbk_conf['log_csv']['log_time']))
        else:
            cbks.append(CSVLogger(output_path=cbk_conf['log_csv']['log_path']
                                  .replace('.csv', str(repetition)+'.csv'),
                                  log_time=cbk_conf['log_csv']['log_time']))

    return cbks


def parse_momentum(mom_conf):
    if mom_conf['use']:
        print('\n\nmom\n\n')
        return Momentum(mom_conf['coeff'])
    else:
        return NoMomentum()


def parse_model(mdl_conf):
    lrs = mdl_conf['layers']
    assert len(lrs) > 0

    layers = [Layer(lrs[0]['nb_neurons'])]

    if len(lrs) > 1:
        for l in lrs[1:]:
            layers.append(Layer(nb_neurons=l['nb_neurons'],
                                activation_function=parse_activation(l['activation']),
                                use_bias=l['bias'],
                                weights_initializer=parse_weights_initializer(l['initializer'])))

    return Model(layers)


def parse_activation(activ_name):

    '''if activ_name == 'Sigmoid':
        return Sigmoid()
    elif activ_name == 'Softmax':
        return Softmax()
    elif activ_name == 'Softplus':
        return Softplus()
    elif activ_name == 'Linear':
        return Linear()
    elif activ_name == 'Elu':
        return Elu()
    elif activ_name == 'Tanh':
        return Tanh()
    else:
        return Sigmoid()'''

    return eval(activ_name+'()')


def parse_weights_initializer(init_name):
    return None if init_name=='default' else eval(init_name)