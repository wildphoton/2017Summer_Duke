import numpy as np
import os
import sys
import scipy.io
import matplotlib.pyplot as plt


from six.moves import xrange

from mat_batch_reader import BatchReader
dataDir = os.path.expanduser("~/Documents/data/igarssTrainingAndTestingData")
matDir = os.path.join(dataDir, "training_patches_for_segmentation")

num_matFiles = range(12, 13)
# num_matFiles = xrange(1, 3)

data_reader = BatchReader(matDir, num_matFiles)
print(np.count_nonzero(data_reader.labels))

mean = data_reader.mean_image
# mean_pixel = np.mean(mean, axis=(0, 1))

print(np.unique(data_reader.annotations))

# image, annotation = data_reader.next_batch(10)
# image = image.astype(float)
# image -= mean
# image, annotation = data_reader.get_random_batch(1)

# np.where(data_reader.annotations == 2)

# plt.imshow(data_reader.annotations[284146,:,:,:])

pass
