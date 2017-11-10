from matplotlib import pyplot as plt
import os
import pandas as pd
import yaml


lbl = {'train_manual_and_augmented':        u'Pełny zbiór uczący + dopisane znaki + dogenerowane dane (5225 obs.)',
       'train_and_manual':                  u'Pełny zbiór uczący + dopisane znaki (1725 obs.)',
       'full_train':                        u'Pełny zbiór uczący (1325 obserwacji)',
       'two_thirds_train':                  u'2/3 zbioru uczącego (887 obs.)',
       'one_third_train':                   u'1/3 zbioru uczącego (437 obs.)',

       'coeff_0_001':                       u'η = 0.001',
       'coeff_0_003':                       u'η = 0.003',
       'coeff_0_01':                        u'η = 0.01',
       'coeff_0_03':                        u'η = 0.03',
       'coeff_0_1':                         u'η = 0.1',
       'coeff_0_3':                         u'η = 0.3',
       'coeff_1':                           u'η = 1.0',

       'momentum_coeff_0_7_tr_coeff_0_1':   u'α = 0.7, η = 0.1',
       'momentum_coeff_0_9_tr_coeff_0_1':   u'α = 0.9, η = 0.1',
       'momentum_coeff_0_9_tr_coeff_0_03':  u'α = 0.9, η = 0.03',
       'no_momentum_tr_coeff_0_1':          u'α = 0.0, η = 0.1',

       'bengio':                            u'Wg. Bengio',
       'default_unif_0_1':                  u'Rozkład jednostajny w ∈ (-0.1; -0.1)',
       'he_normal_bengio_adjusted':         u'Dostosowane do architektury - He (normal) + Bengio',
       'he_uniform_bengio_adjusted':        u'Dostosowane do architektury - He (uniform) + Bengio',
       'lecture':                           u'Metoda proponowana na wykładzie'

       }
'''
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
    '''

def plot_learn_coeff():
    experiment = 'learn_coeff_tests'
    dfs = []
    for setting in os.listdir('../../logs/%s/' % experiment):
        if '.yaml' not in setting:
            dfs.append((setting, pd.read_csv('../../logs/%s/%s/summary.csv' % (experiment, setting))))
    for df in dfs:
        l = plt.errorbar(df[1].Epoch.iloc[1:], df[1].ValLoss_mean.iloc[1:],
                         yerr=df[1].ValLoss_std.iloc[1:],
                         label=lbl[df[0]])

    plt.legend(fontsize='xx-large')
    plt.title('Wartość funkcji straty na zbiorze walidacyjnym w kolejnych epokach nauki dla różnych wartości współczynnika uczenia', fontsize='xx-large')
    plt.ylabel('Wartość funkcji straty (crossentropy)', fontsize='xx-large')
    plt.xlabel('Nr epoki', fontsize='xx-large')
    plt.show()


def plot_momentum():
    experiment = 'momentum_tests'
    dfs = []
    for setting in os.listdir('../../logs/%s/' % experiment):
        if '.yaml' not in setting:
            dfs.append((setting, pd.read_csv('../../logs/%s/%s/summary.csv' % (experiment, setting))))
    for df in dfs:
        l = plt.errorbar(df[1].Epoch.iloc[1:], df[1].ValLoss_mean.iloc[1:],
                         yerr=df[1].ValLoss_std.iloc[1:],
                         label=lbl[df[0]])

    plt.legend(fontsize='xx-large')
    plt.title('Wartość funkcji straty na zbiorze walidacyjnym w kolejnych epokach nauki dla różnych konfiguracji momentum', fontsize='xx-large')
    plt.ylabel('Wartość funkcji straty (crossentropy)', fontsize='xx-large')
    plt.xlabel('Nr epoki', fontsize='xx-large')
    plt.show()


def plot_weights():
    experiment = 'weights_init_tests'
    dfs = []
    for setting in os.listdir('../../logs/%s/' % experiment):
        if '.yaml' not in setting:
            dfs.append((setting, pd.read_csv('../../logs/%s/%s/summary.csv' % (experiment, setting))))
    for df in dfs:
        l = plt.errorbar(df[1].Epoch.iloc[1:], df[1].ValLoss_mean.iloc[1:],
                         yerr=df[1].ValLoss_std.iloc[1:],
                         label=lbl[df[0]])

    plt.legend(fontsize='xx-large')
    plt.title('Wartość funkcji straty na zbiorze walidacyjnym w kolejnych epokach nauki dla różnych metod inicjalizacji wag', fontsize='xx-large')
    plt.ylabel('Wartość funkcji straty (crossentropy)', fontsize='xx-large')
    plt.xlabel('Nr epoki', fontsize='xx-large')
    plt.show()


def plot_data_size():
    experiment = 'dataset_size_tests'
    dfs = []
    #for setting in os.listdir('../../logs/%s/' % experiment):
    for setting in ['one_third_train', 'two_thirds_train', 'full_train',
                    'train_and_manual', 'train_manual_and_augmented']:
        if '.yaml' not in setting:
            dfs.append((setting, pd.read_csv('../../logs/%s/%s/summary.csv' % (experiment, setting))))
    for df in dfs:
        l = plt.errorbar(df[1].Epoch.iloc[1:], df[1].ValLoss_mean.iloc[1:],
                         yerr=df[1].ValLoss_std.iloc[1:],
                         label=lbl[df[0]])

    plt.legend(fontsize='xx-large')
    plt.title('Wartość funkcji straty na zbiorze walidacyjnym w kolejnych epokach nauki dla różnych rozmiarów zbioru uczącego', fontsize='xx-large')
    plt.ylabel('Wartość funkcji straty (crossentropy)', fontsize='xx-large')
    plt.xlabel('Nr epoki', fontsize='xx-large')
    plt.show()


def plot_learning_time():

    def autolabel(ax, bars, labels, errs):
        for bar, lbl, err in zip(bars, labels, errs):
            height = bar.get_height()
            print(bar.get_x())
            ax.text(bar.get_x() + bar.get_width() / 2., height+err+1 + (2 if bar.get_x()+bar.get_width()/2.==35 else 0),
                   '{:.1f} ± {:.1f}'.format(lbl, err),
                   ha='center', va='bottom',
                    fontsize='x-large')

    experiment = 'nb_hidden_neurons_tests'
    fig, ax = plt.subplots()
    df = pd.read_csv('../../logs/%s/results.csv'%experiment)
    l = ax.bar(df['size'], df.epochs_mean, yerr=df.epochs_std, width=4)#,
                         #label=lbl[])

    autolabel(ax, l, df.epochs_mean, df.epochs_std)

    ax.set_xticks(df['size'])
    ax.set_xticklabels([str(x) for x in df['size']])
    plt.title('Liczba epok potrzebnych do osiągnięcia skuteczności (accuracy) > 85% przy różnych rozmiarach warstwy ukrytej', fontsize='xx-large')
    plt.ylabel('Liczba epok', fontsize='xx-large')
    plt.xlabel('Liczba neuronów w warstwie ukrytej', fontsize='xx-large')
    plt.show()



#plot_learn_coeff()
#plot_momentum()
plot_weights()
#plot_data_size()
#plot_learning_time()

def table_to_tex(logdf, path):
    logdf.to_csv(path, sep='&', line_terminator=' \\\\\n', index=False)

#df = pd.read_csv('../../logs/nb_hidden_neurons_tests/results.csv')

#print(df[['size', 'epochs_mean', 'epochs_std']])
#table_to_tex(df[['size', 'epochs_mean', 'epochs_std']], '../../logs/nb_hidden_neurons_tests/results.tex')

