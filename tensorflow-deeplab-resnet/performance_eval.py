"""
evaluate the mean IoU and PR curve of prediction on large images
"""
import scipy.misc as misc
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

sys.path.append(os.path.realpath('../metrics'))
from eval_segm import *
from seg_metric import SegMetric

imageFileName = ['11ska595800{}','11ska460755{}']
num_val = imageFileName.__len__()

truthDir = os.path.expanduser('~/Documents/data/igarssTrainingAndTestingData/imageFiles')

learning_rates = [1e-3]
weight_decays = [0.0005]
batch_sizes = [20]
for weight_decay in weight_decays:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            if (batch_size != 20) | (learning_rate != 1e-5) | (weight_decay != 0.0005):
                validDir = os.path.join(os.path.realpath('./snapshots'), 'with_pretrained_model/sc_batchsize{}_learningRate_{:.0e}_weight_decay_{}/'.format(batch_size, learning_rate, weight_decay),'images')
                seg_metric = SegMetric(1)
                for i in range(0, num_val):
                    image = misc.imread(os.path.join(truthDir,imageFileName[i].format('.png')))
                    truth = (misc.imread(os.path.join(truthDir,imageFileName[i].format('_truth.png')))/255).astype(np.uint8)
                    valid_pmap = misc.imread(os.path.join(validDir,imageFileName[i].format('_valid_float.tif')))
                    pred_binary = (valid_pmap>0.5).astype(np.uint8)
                    seg_metric.add_image_pair(pred_binary, truth)
                print('sc_batchsize{}_learningRate_{:.0e}_weight_decay_{}'.format(batch_size, learning_rate, weight_decay) + " mean_IU: {}".format(seg_metric.mean_IU()))