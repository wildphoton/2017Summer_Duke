"""
This script is for unifying resolution of all large building images cross different cities into 1 feet
"""
import scipy.misc as misc
import os
import numpy as np
import sys
import glob

from patchify import patchify

citylist = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
resolutionList = {'Norfolk': 1.0, 'Arlington': 0.984252, 'Atlanta': 0.5, 'Austin': 0.5, 'Seekonk': 0.984252,
                  'NewHaven': 0.984252}
# resolution of data at different cities (in foot)
dataDir = os.path.expanduser("~/Documents/data/building")

for city_name in citylist:
    imageDir = os.path.join(dataDir, '{}'.format(city_name))
    num_image = glob.glob(os.path.join(imageDir, "{}_*_RGB.png".format(city_name))).__len__()
    patch_size = 128

    for j in range(1, 4):
        if j == 3 and city_name == 'NewHaven':
            break

        imageFile = os.path.join(imageDir, "{}_{:0>2}_RGB.png".format(city_name, j))
        truthFile = os.path.join(imageDir, "{}_{:0>2}_truth.png".format(city_name, j))

        image_large = misc.imread(imageFile)
        truth_large = (misc.imread(truthFile))

        # resize images to resolution as 1 feet
        image_large = misc.imresize(image_large, resolutionList[city_name], interp='bilinear')
        truth_large = misc.imresize(truth_large, resolutionList[city_name], interp='nearest')

        imageFile_1feet = os.path.join(imageDir, "{}_{:0>2}_RGB_1feet.png".format(city_name, j))
        truthFile_1feet = os.path.join(imageDir, "{}_{:0>2}_truth_1feet.png".format(city_name, j))

        misc.imsave(imageFile_1feet, image_large)
        misc.imsave(truthFile_1feet, truth_large)
