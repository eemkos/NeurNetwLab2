import Augmentor

from src.preprocessing.augmentations import PadToSize
from src.preprocessing.augmentations import ElasticTransform
from src.preprocessing.augmentations import Shift, PoissonNoise


SOURCE_DIR = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2/data/train/%d/'
OUTPUT_DIR = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2/data/augmented/%d/'


def perform_augmentations():
    augs_dict = {
        0: [(zoom_out_and_shift, 100), (elastic, 75), (elastic_scaled, 125), (shift, 50)],
        1: [(zoom_out_and_shift, 150), (elastic, 125), (shift, 75)],
        2: [(zoom_out_and_shift, 150), (elastic, 125), (shift, 75)],
        3: [(zoom_out_and_shift, 150), (elastic, 125), (shift, 75)],
        4: [(elastic, 175), (shift, 125), (elastic_scaled, 50)],
        5: [(zoom_out_and_shift, 100), (elastic, 175), (shift, 75)],
        6: [(zoom_out_and_shift, 100), (elastic, 75), (elastic_scaled, 125), (shift, 50)],
        7: [(zoom_out_and_shift, 100), (elastic, 75), (elastic_scaled, 125), (shift, 50)],
        8: [(zoom_out_and_shift, 100), (elastic, 75), (elastic_scaled, 125), (shift, 50)],
        9: [(zoom_out_and_shift, 100), (elastic, 75), (elastic_scaled, 125), (shift, 50)]
    }

    for k in augs_dict:
        augs = augs_dict[k]
        augs_dict[k] = [(aug[0], 3*aug[1]) for aug in augs]

    #augs_dict = {
    #    0: [(poisson, 3)],
    #    1: [(poisson, 3)],
    #    2: [(poisson, 3)],
    #    3: [(poisson, 3)],
    #    4: [(poisson, 3)],
    #    5: [(poisson, 3)],
    #    6: [(poisson, 3)],
    #    7: [(poisson, 3)],
    #    8: [(poisson, 3)],
    #    9: [(poisson, 3)],
    #}

    for i in range(10):
        for aug in augs_dict[i]:
            #aug[0](i, int(aug[1]/10))
            aug[0](i, aug[1])


def zoom_out_and_shift(number, nb_samples=10):
    ppln = Augmentor.Pipeline(source_directory=SOURCE_DIR % number, output_directory=OUTPUT_DIR % number, save_format='png')

    ppln.resize(probability=1, width=5, height=7)
    ppln.add_operation(PadToSize(probability=1, width=7, height=10))
    ppln.add_operation(Shift(probability=1, max_shift_horizontal=2, max_shift_vertical=3))
    ppln.zoom_random(probability=0.5, percentage_area=0.9)
    ppln.add_operation(PoissonNoise(probability=0.5, min_peak=0.3, max_peak=1.0))


    ppln.sample(nb_samples)


def elastic_scaled(number, nb_samples=10):
    ppln = Augmentor.Pipeline(source_directory=SOURCE_DIR % number, output_directory=OUTPUT_DIR % number, save_format='png')

    ppln.resize(probability=1, width=5, height=7)
    ppln.add_operation(PadToSize(probability=1, width=7, height=10))
    ppln.add_operation(ElasticTransform(probability=1, alpha=0.1, sigma=0.1, alpha_affine=0.00001))
    ppln.zoom_random(probability=0.25, percentage_area=0.9)
    ppln.add_operation(PoissonNoise(probability=0.5, min_peak=0.3, max_peak=1.0))
    ppln.sample(nb_samples)


def elastic(number, nb_samples=10):
    ppln = Augmentor.Pipeline(source_directory=SOURCE_DIR % number, output_directory=OUTPUT_DIR % number, save_format='png')

    ppln.add_operation(ElasticTransform(probability=1, alpha=0.1, sigma=0.1, alpha_affine=0.00001))
    ppln.add_operation(PoissonNoise(probability=0.5, min_peak=0.3, max_peak=1.0))

    ppln.sample(nb_samples)


def shift(number, nb_samples=10):
    ppln = Augmentor.Pipeline(source_directory=SOURCE_DIR % number, output_directory=OUTPUT_DIR % number, save_format='png')

    ppln.add_operation(Shift(probability=1, max_shift_horizontal=1, max_shift_vertical=1))
    ppln.add_operation(PoissonNoise(probability=0.5, min_peak=0.3, max_peak=1.0))

    ppln.sample(nb_samples)


def poisson(number, nb_samples=10):
    ppln = Augmentor.Pipeline(source_directory=SOURCE_DIR % number, output_directory=OUTPUT_DIR % number, save_format='png')

    ppln.add_operation(PoissonNoise(probability=1, min_peak=0.3, max_peak=1.0))
    ppln.sample(nb_samples)

perform_augmentations()