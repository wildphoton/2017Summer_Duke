import numpy as np
import os
import sys
import scipy.io
import scipy.misc as misc
import matplotlib.pyplot as plt


from six.moves import xrange

from mat_batch_reader import BatchReader
dataDir = os.path.expanduser("~/Documents/data/igarssTrainingAndTestingData")
matDir = os.path.join(dataDir, "training_patches_for_segmentation")
trainDir = os.path.join(dataDir, "training_patches_images_size41_all")
num_matFiles = range(1, 13)
# num_matFiles = xrange(1, 3)

data_reader = BatchReader(matDir, num_matFiles)

mean = data_reader.mean_image
# print(np.count_nonzero(data_reader.labels))
mean_pixel = np.mean(mean, axis=(0, 1))

# print(np.unique(data_reader.annotations))

# image, annotation = data_reader.next_batch(10)
# image = image.astype(float)
# image -= mean
# image, annotation = data_reader.get_random_batch(1)

# np.where(data_reader.annotations == 2)

# plt.imshow(data_reader.annotations[284146,:,:,:])



'''write image patches into jpg/png files'''
if not os.path.isdir(trainDir):
    os.makedirs(trainDir)
print(data_reader.images.shape[0])
file_path = os.path.expanduser("~/Documents/zhenlinx/code/tensorflow-deeplab-resnet/dataset")
with open(os.path.join(file_path, 'train_sc_all.txt'),'w') as file:
    for i in range(data_reader.images.shape[0]): #data_reader.images.shape[0]
        # get image/truth patches
        image_temp = data_reader.images[i]
        truth_temp = data_reader.annotations[i,:,:,0]
        image_name_temp = "/image_{:0>6}.jpg".format(i)
        truth_name_temp = "/truth_{:0>6}.png".format(i)
        file.write(image_name_temp+" "+truth_name_temp + '\n')
        misc.imsave(trainDir + image_name_temp, image_temp)
        misc.imsave(trainDir + truth_name_temp, truth_temp)
pass
