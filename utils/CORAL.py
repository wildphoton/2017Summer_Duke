import numpy as np
import scipy.misc as misc
import scipy.stats as stats
import scipy.linalg as linalg
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.realpath('../utils'))
from patchify import patchify


def CORAL(source_data, target_data, lamda):
    """
    CORAL domain adaptation
    :param source_data: M_s by N_s matrix, M_s is number of samples and N_s is the dimension of features
    :param target_data: M_t by N_t matrix, M_t is number of samples and N_t is the dimension of features
    :return: the source samples that adapted to target domain: M_s by N_t matrix
    """

    cov_s = np.cov(source_data, rowvar=False) + lamda * np.eye(np.shape(source_data)[1])
    cov_t = np.cov(target_data, rowvar=False) + lamda * np.eye(np.shape(target_data)[1])
    source_data_whiten = np.matmul(source_data, linalg.fractional_matrix_power(cov_s, -0.5))
    source_data_colorize = np.matmul(source_data_whiten, linalg.fractional_matrix_power(cov_t, 0.5))

    return source_data_colorize


def image_adapt(source_image, target_image, lamda):
    # zero-mean
    source_mean = np.mean(source_image, (0, 1))
    target_mean = np.mean(target_image, (0, 1))

    source_image_vec = np.reshape(source_image, [-1, 3]) - source_mean
    target_image_vec = np.reshape(target_image, [-1, 3]) - target_mean

    source_image_adapted_vec = CORAL(source_image_vec, target_image_vec, lamda)

    source_image_adapted_vec += target_mean

    return np.reshape(source_image_adapted_vec, source_image.shape).astype(np.uint8)


source_city_name = "Arlington"
target_city_name = "Norfolk"

IMAGE_PATH = os.path.expanduser("~/Documents/data/building")
source_image_file = '{}_{:0>2}'.format(source_city_name, 1)
source_image = misc.imread(os.path.join(IMAGE_PATH, source_city_name, "{}_RGB.png".format(source_image_file)))
source_truth = (
    misc.imread(os.path.join(IMAGE_PATH, source_city_name, "{}_truth.png".format(source_image_file))) / 255).astype(
    np.uint8)

target_image_file = '{}_{:0>2}'.format(target_city_name, 1)
target_image = misc.imread(os.path.join(IMAGE_PATH, target_city_name, "{}_RGB.png".format(target_image_file)))
target_truth = (
    misc.imread(os.path.join(IMAGE_PATH, target_city_name, "{}_truth.png".format(target_image_file))) / 255).astype(
    np.uint8)

LAMBDA = 1
source_image_adapted = image_adapt(source_image, target_image, LAMBDA)
