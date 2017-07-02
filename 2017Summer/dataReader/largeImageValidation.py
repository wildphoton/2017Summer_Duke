import scipy.misc as misc
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

sys.path.append(os.path.realpath('../metrics'))
from eval_segm import *

patchsize = 128
imageFileName = ['11ska595800{}','11ska460755{}']
truthDir = os.path.expanduser('~/Documents/data/igarssTrainingAndTestingData/imageFiles')
validDir = os.path.join(os.path.realpath('../FCN/logs'), 'logs_batch%d' % patchsize,'images')

i=1
image = misc.imread(os.path.join(truthDir,imageFileName[i].format('.png')))
truth = (misc.imread(os.path.join(truthDir,imageFileName[i].format('_truth.png')))/255).astype(np.uint8)
valid_pmap = misc.imread(os.path.join(validDir,imageFileName[i].format('_valid_float.tif')))


precision, recall, thresholds = precision_recall_curve(truth.flatten(), valid_pmap.flatten(), 1)

print("mean_IU: {}".format(mean_IU((valid_pmap > 0.5).astype(int), truth)))

# Plot PR curve
plt.figure()
plt.plot(recall, precision, lw=2, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recal')
# plt.legend(loc="lower left")
plt.show()


pass
