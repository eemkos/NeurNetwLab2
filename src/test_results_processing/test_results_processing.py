import pandas as pd
import os


DIRPATH = '../../logs/momentum_tests/momentum_coeff_0_9_tr_coeff_0_1/'


def summarise_results(dirpath):
    dfs = []

    for i in range(10):
        dfs.append(pd.read_csv(dirpath+'trial%d.csv'%i))

    df = pd.concat(dfs)
    summarise = df.groupby('Epoch').agg([pd.DataFrame.mean, pd.DataFrame.std]).reset_index()
    summarise.columns = ['_'.join(col).strip().strip('_') for col in summarise.columns.values]
    summarise.to_csv(dirpath+'summary.csv', index=False)


'''
for experiment in os.listdir('../../logs/'):
    for setting in os.listdir('../../logs/%s/'%experiment):
        summarise_results('../../logs/%s/%s/' % (experiment, setting))
'''

summarise_results('../../logs/%s/%s/' % ('learn_coeff_tests', 'coeff_0_001'))

