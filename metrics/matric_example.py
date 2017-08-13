import scipy.misc as misc
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from eval_segm import *
from seg_metric import SegMetric

imageFileName = ['11ska595800{}', '11ska460755{}', '11ska580860{}', '11ska565845{}']
num_val = imageFileName.__len__()
truthDir = os.path.expanduser('~/Documents/data/igarssTrainingAndTestingData/imageFiles')

batch_size = 128
validDir = os.path.join(os.path.realpath('../FCN/logs'), 'logs_batch%d' % batch_size, 'images')

seg_metric = SegMetric(1)
# plt.figure("PR Curve")
for i in range(0, num_val):
    image = misc.imread(os.path.join(truthDir, imageFileName[i].format('.png')))
    truth = (misc.imread(os.path.join(truthDir, imageFileName[i].format('_truth.png'))) / 255).astype(np.uint8)
    valid_pmap = misc.imread(os.path.join(validDir, imageFileName[i].format('_valid_pmap.tif')))
    pred_binary = (valid_pmap > 0.5).astype(np.uint8)

    # mean IoU
    seg_metric.add_image_pair(pred_binary, truth)
    print("Image {}: {:.4f}".format(imageFileName[i][0:-2], mean_IU(pred_binary, truth)))

    # F1 score
    print("F1 {}".format(f1_score(truth, pred_binary, average='micro')))

    # RP curve
    # Plot PR curve
    # plt.figure("PR Curve")
    # precision, recall, thresholds = precision_recall_curve(truth.flatten(), valid_pmap.flatten(), 1)
    #
    # plt.plot(recall, precision, lw=2,
    #          label=imageFileName[i][0:-2])
    # plt.legend()
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recal')

print("Overal mean IoU: {:.4f}".format(seg_metric.mean_IU()))