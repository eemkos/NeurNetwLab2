from matplotlib import pyplot as plt
import os
import pandas as pd
import yaml


lbl = {'train_manual_and_augmented':        u'Pełny zbiór uczący + dopisane znaki + dogenerowane dane (5225)',
       'train_and_manual':                  u'Pełny zbiór uczący + dopisane znaki (1725)',
       'full_train':                        u'Pełny zbiór uczący (1325)',
       'two_thirds_train':                  u'2/3 zbioru uczącego (887)',
       'one_third_train':                   u'1/3 zbioru uczącego (437)',

       'coeff_0_001':                       u'η = 0.001',
       'coeff_0_003':                       u'η = 0.003',
       'coeff_0_01':                        u'η = 0.01',
       'coeff_0_03':                        u'η = 0.03',
       'coeff_0_1':                         u'η = 0.1',
       'coeff_0_3':                         u'η = 0.3',
       'coeff_1':                           u'η = 1.0',

       'momentum_coeff_0_7_tr_coeff_0_1':   u'α = 0.7, η = 0.1',
       'momentum_coeff_0_9_tr_coeff_0_1':   u'α = 0.9, η = 0.1',
       'no_momentum_tr_coeff_0_1':          u'α = 0.0, η = 0.1',

       'bengio':                            u'Wg. Bengio',
       'default_unif_0_1':                  u'Rozkład jednostajny w ∈ (-0.1; -0.1)',
       'he_normal_bengio_adjusted':         u'Dostosowane do architektury - He (normal) + Bengio',
       'he_uniform_bengio_adjusted':        u'Dostosowane do architektury - He (uniform) + Bengio',
       'lecture':                           u'Metoda proponowana na wykładzie'

       }

i = 1
for experiment in os.listdir('../../logs/'):
    dfs = []
    for setting in os.listdir('../../logs/%s/'%experiment):
        if '.yaml' not in setting:
            dfs.append((setting, pd.read_csv('../../logs/%s/%s/summary.csv' % (experiment, setting))))
    plt.figure(i)
    i += 1
    for df in dfs:

        l = plt.errorbar(df[1].Epoch.iloc[1:], df[1].ValLoss_mean.iloc[1:],
                         yerr=df[1].ValLoss_std.iloc[1:],
                         label=lbl[df[0]])

    plt.legend()
    #plt.show(block=False)


plt.show()