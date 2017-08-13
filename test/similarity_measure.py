import numpy as np
import scipy.misc as misc
import os
import sys
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim


source_city_name = "Seekonk"
target_city_name = "Seekonk"

IMAGE_PATH = os.path.expanduser("~/Documents/data/building")
source_image_file = '{}_{:0>2}'.format(source_city_name, 1)
source_image = misc.imread(os.path.join(IMAGE_PATH, source_city_name, "{}_RGB.png".format(source_image_file))).astype(np.uint8)

target_image_file = '{}_{:0>2}'.format(target_city_name, 1)
target_image = misc.imread(os.path.join(IMAGE_PATH, target_city_name, "{}_RGB.png".format(target_image_file))).astype(np.uint8)

ssim = compare_ssim(misc.imresize(source_image, target_image.shape), target_image, multichannel=True)
print(ssim)