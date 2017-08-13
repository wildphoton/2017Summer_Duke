import scipy.misc as misc
import os
import numpy as np
import sys
from patchify import patchify

imageFileName = ['11ska595800{}', '11ska460755{}']
dataDir = os.path.expanduser("~/Documents/data/igarssTrainingAndTestingData")
imageDir = os.path.join(dataDir, 'imageFiles')
validDir = os.path.join(dataDir, 'validation_patches_images_size41')

i = 0
valid_image_large = misc.imread(os.path.join(imageDir, imageFileName[i].format('.png')))
valid_truth_large = (misc.imread(os.path.join(imageDir, imageFileName[i].format('_truth.png'))) / 255).astype(np.uint8)

valid_stride = 20
IMAGE_SIZE = 41

valid_image_patches = patchify(valid_image_large, IMAGE_SIZE, valid_stride)
valid_truth_patches = patchify(valid_truth_large, IMAGE_SIZE, valid_stride)

'''write image patches into jpg/png files'''
if not os.path.isdir(validDir):
    os.makedirs(validDir)

file_path = os.path.expanduser("~/Documents/zhenlinx/code/tensorflow-deeplab-resnet/dataset")
with open(os.path.join(file_path, 'val_sc.txt'), 'w') as file:
    for i in range(valid_image_patches.shape[0]):  # data_reader.images.shape[0]
        # get image/truth patches
        image_temp = valid_image_patches[i]
        truth_temp = valid_truth_patches[i]
        image_name_temp = "/image_{:0>6}.jpg".format(i)
        truth_name_temp = "/truth_{:0>6}.png".format(i)
        file.write(image_name_temp + " " + truth_name_temp + '\n')
        misc.imsave(validDir + image_name_temp, image_temp)
        misc.imsave(validDir + truth_name_temp, truth_temp)
pass
