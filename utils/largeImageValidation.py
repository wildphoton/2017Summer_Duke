# mean IoU for ROIs
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

batch_size = 128
learning_rate = 1e-3

imageFileName = ['11ska595800{}', '11ska460755{}', '11ska580860{}', '11ska565845{}']
num_val = imageFileName.__len__()

truthDir = os.path.expanduser('~/Documents/data/igarssTrainingAndTestingData/imageFiles')
validDir = [os.path.join(os.path.realpath('../FCN/logs'), 'logs_batch%d' % batch_size, 'images')]
learning_rates = [1e-5, 1e-4, 1e-3]
weight_decay = 0.0005
batch_sizes = [20]
for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        if learning_rate == 1e-4:
            batch_size = 40
        validDir.append(
            os.path.join(os.path.realpath('../../tensorflow-deeplab-resnet/snapshots/train_with_pretrained_model'),
                         'sc_all_batchsize{}_learningRate_{:.0e}_weight_decay_{}/'.format(batch_size, learning_rate,
                                                                                          weight_decay), 'images'))
pred_binary = []

seg_metric = SegMetric(1)

meanIoUs = []
mean_performance = []
variance_performance = []
patch_size = 20
for i in range(0, 1):
    image = misc.imread(os.path.join(truthDir, imageFileName[i].format('.png')))
    truth = (misc.imread(os.path.join(truthDir, imageFileName[i].format('_truth.png'))) / 255).astype(np.uint8)
    region_center = np.nan_to_num(center_of_mass(truth, label(truth)[0], range(label(truth)[1])))
    meanIoU = np.zeros([region_center.__len__() - 1, validDir.__len__()])

    for l in range(validDir.__len__()):
        pred_binary.append(
            ((misc.imread(os.path.join(validDir[l], imageFileName[i].format('_valid_pmap.tif')))) > 0.5).astype(int))

    for j in range(1, region_center.shape[0]):
        left_up = np.max(np.array([region_center[j] - patch_size, np.array([0, 0])]), axis=0).astype(int)
        right_bottom = np.min(np.array([region_center[j] + patch_size, np.array([5000, 5000])]), axis=0).astype(int)
        image_patch = image[left_up[0]:right_bottom[0], left_up[1]:right_bottom[1]]
        truth_patch = truth[left_up[0]:right_bottom[0], left_up[1]:right_bottom[1]]
        for k in range(validDir.__len__()):
            pred_patch = pred_binary[k][left_up[0]:right_bottom[0], left_up[1]:right_bottom[1]]
            meanIoU[j - 1, k] = mean_IU(pred_patch, truth_patch)
    mean_performance.append(np.mean(meanIoU, axis=1))
    variance_performance.append(np.var(meanIoU, axis=1))
    meanIoUs.append(meanIoU)

for i in range(0, 1):
    image = misc.imread(os.path.join(truthDir, imageFileName[i].format('.png')))
    truth = (misc.imread(os.path.join(truthDir, imageFileName[i].format('_truth.png'))) / 255).astype(np.uint8)
    region_center = np.nan_to_num(center_of_mass(truth, label(truth)[0], range(label(truth)[1])))

    mean_ind = np.argsort(mean_performance[0])
    var_ind = np.argsort(variance_performance[0])

    num_sample = 3
    for j in range(0, num_sample):
        left_up = np.max(np.array([region_center[mean_ind[j]] - patch_size, np.array([0, 0])]), axis=0).astype(int)
        right_bottom = np.min(np.array([region_center[mean_ind[j]] + patch_size, np.array([5000, 5000])]),
                              axis=0).astype(int)
        image_patch = image[left_up[0]:right_bottom[0], left_up[1]:right_bottom[1]]
        truth_patch = truth[left_up[0]:right_bottom[0], left_up[1]:right_bottom[1]]
        # plt.subplot(num_sample, truthDir.__len__() + 2, (truthDir.__len__() + 2) * j + 1)
        # plt.imshow(image_patch)
        # plt.subplot(num_sample, truthDir.__len__() + 2, (truthDir.__len__() + 2) * j + 2)
        # plt.imshow(truth_patch)
        for k in range(validDir.__len__()):
            pred_patch = pred_binary[k][left_up[0]:right_bottom[0], left_up[1]:right_bottom[1]]
            # plt.subplot(num_sample, truthDir.__len__() + 2, (truthDir.__len__() + 2) * j + 2 + k)
            # plt.imshow(pred_patch)

pass
