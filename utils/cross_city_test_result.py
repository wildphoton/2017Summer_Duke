import scipy.misc as misc
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from scipy.ndimage.measurements import *
from skimage import measure

sys.path.append(os.path.realpath('../metrics'))
from eval_segm import *
from seg_metric import SegMetric

train_city_list = ['Norfolk', 'Arlington', 'Austin', 'NewHaven']
test_city_list = ['Norfolk']
learning_rate = 1e-3
batch_size = 20
input_size = '128,128'
decay_step = 10  # in epochs
decay_rate = 0.1
IMAGE_SIZE = 128
weight_decay = 0.0005
ind = 1  # training image index
for test_city_name in test_city_list:
    IMAGE_PATH = os.path.expanduser("~/Documents/data/building/{}".format(test_city_name))
    test_image_file = '{}_{:0>2}'.format(test_city_name, 2 if test_city_name == 'NewHaven' else 3)
    test_image = misc.imread(os.path.join(IMAGE_PATH, "{}_RGB.png".format(test_image_file)))
    test_truth = (misc.imread(os.path.join(IMAGE_PATH, "{}_truth.png".format(test_image_file))) / 255).astype(np.uint8)
    pred_NT = {}
    pred_T = {}
    # plt.figure('Testing on {}'.format(test_image_file), figsize=(120, 10))
    f, axes = plt.subplots(1, num_train * 2 + 2, figsize=(10, 80))

    axes[0].imshow(test_image)
    axes[0].set_title('{} image'.format(test_image_file))

    axes[1].imshow(test_truth)
    axes[1].set_title('{} truth'.format(test_truth))

    for i, train_city_name in enumerate(train_city_list):
        snapshot_path = os.path.expanduser(
            "~/Documents/zhenlinx/code/tensorflow-deeplab-resnet/snapshots_building/train_with_pretrained_model/{}_{:0>2}_batchsize{}_learningRate_{:.0e}_L2weight_{}_decayStep_{:d}_decayRate{}".format(
                train_city_name, ind, batch_size, learning_rate, weight_decay, decay_step, decay_rate))

        image_path_NT = os.path.join(snapshot_path, 'images_NT')
        pred_NT[train_city_name] = misc.imread(
            os.path.join(image_path_NT, '{}_valid_pred_255.png'.format(test_image_file)))

        axes[2 * (i + 1)].imshow(pred_NT[train_city_name])
        axes[2 * (i + 1)].set_title('{}_{}_NT'.format(test_image_file, train_city_name))

        image_path_T = os.path.join(snapshot_path, 'images')
        pred_T[train_city_name] = misc.imread(
            os.path.join(image_path_T, '{}_valid_pred_255.png'.format(test_image_file)))

        axes[2 * (i + 1) + 1].imshow(pred_T[train_city_name])
        axes[2 * (i + 1) + 1].set_title('{}_{}_T'.format(test_image_file, train_city_name))

    plt.show()
