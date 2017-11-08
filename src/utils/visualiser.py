'''from src.config import config
from drawnow import drawnow
from functools import partial
from matplotlib import pyplot as plt
import numpy as np

plt.ion()


def visualise_on_plot(weights, is_last=False):
    drawnow(partial(visualise, weights=weights, is_last=is_last))


def visualise(weights, is_last=False):
    from src.config import config

    def linear_func_x1_to_x2(x):
        if config['use_bias']:
            w0 = weights[0]
            w1 = weights[1]
            w2 = weights[2]
        else:
            w0 = config['theta']
            w1 = weights[0]
            w2 = weights[1]
        return (-w0 - w1 * x) / w2

    zero = -1.0 if config['activation'] == 'bipolar' else 0.0
    one = 1.0

    combinations = [(zero, zero), (one, zero), (zero, one), (one, one)]

    x1 = np.linspace(-5, 5)
    x2 = np.array([linear_func_x1_to_x2(x) for x in x1])
    plt.axis([-2, 2, -2, 2])
    plt.plot(x1, x2)
    for combination in combinations:
        color = 'b' if combination[1] <= linear_func_x1_to_x2(combination[0]) else 'r'
        plt.plot(*combination, color + 'o')


    plt.show(block=is_last)
'''

