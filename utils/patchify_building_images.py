"""
Sample small patches from large images of building dataset with functions in patchify.py
There are two ways:
    1. Sample with a mesh grid
    2. Sample randomly with balancing the ratio of background and foreground
"""
import scipy.misc as misc
import os
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt

from patchify import patchify_balanced, patchify

citylist = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
resolutionList = {'Norfolk': 1.0, 'Arlington': 0.984252, 'Atlanta': 0.5, 'Austin': 0.5, 'Seekonk': 0.984252,
                  'NewHaven': 0.984252}
# resolution of data at different cities (in foot)
dataDir = os.path.expanduser("~/Documents/data/building")

image_mean_list = {'Norfolk': [127.07435926, 129.40160709, 128.28713284],
                   'Arlington': [88.30304996, 94.97338776, 93.21268212],
                   'Atlanta': [101.997014375, 108.42171833, 110.044871],
                   'Austin': [97.0896012682, 102.94697026, 100.7540157],
                   'Seekonk': [86.67800904, 93.31221168, 92.1328146],
                   'NewHaven': [106.7092798, 111.4314, 110.74903832]}
image_std_list = {'Norfolk': [28.615469420031832, 32.662536832452886, 37.64149854207523],
                  'Arlington': [30.40903039206398, 37.973725024862595, 43.58402191634698],
                  'Atlanta': [36.93662467838125, 39.43470059838385, 41.74732676809388],
                  'Austin': [42.81337177109884, 43.71071321350751, 44.440517007230675],
                  'Seekonk': [25.506449467410715, 32.46885262572024, 37.76814267502958],
                  'NewHaven': [33.05784541012469, 36.62685162291547, 37.686084270914435]}

dataDir = os.path.expanduser("~/Documents/data/building")

citylist = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
for city_name in citylist:
    imageDir = os.path.join(dataDir, '{}'.format(city_name))

    patch_size = 128

    ratio_foreground = 0.5  # ratio of patches with the foreground class
    foreground_threshold = 0.1  # threshold of ratio of true pixels that a patch is classified as foreground
    patch_shape = (patch_size, patch_size)

    for j in range(1, 3):
        if j == 1:  # use the first large image as training data
            patch_stride = int(patch_size / 4)  # training patch stride size is a quarter of the patch size
            num_patch = 20000  # number of patches to generate
        elif j == 2:
            patch_stride = int(patch_size)  # validation patch stride size is the patch size
            num_patch = 2000  # number of patches to generate

        imageFile_1feet = os.path.join(imageDir, "{}_{:0>2}_RGB_1feet.png".format(city_name, j))
        truthFile_1feet = os.path.join(imageDir, "{}_{:0>2}_truth_1feet.png".format(city_name, j))

        image_large = misc.imread(imageFile_1feet)
        truth_large = (misc.imread(truthFile_1feet) / 255).astype(np.uint8)

        # measure the statistics of data
        # image_mean = np.mean(image_large, axis=(0,1))
        # print("Image mean of {} in BGR order: [{}, {}, {}]".format(imageFile, image_mean[2], image_mean[1], image_mean[0]))
        # image_std = np.std(image_large, axis=(0,1))
        # print("Image std of {} in BGR order: [{:3}, {}, {}]".format(imageFile, image_std[2], image_std[1], image_std[0]))


        print("Generating patches of {}_{:0>2}".format(city_name, j))
        # randomly and balanced sampling
        image_patches, truth_patches = patchify_balanced(image_large, truth_large, num_patch, patch_shape,
                                                         ratio_foreground=0.5, foreground_threshold=0.1)

        # sampling with a mesh grid
        # image_patches = patchify(image_large, patch_size, patch_stride)
        # truth_patches = patchify(truth_large, patch_size, patch_stride)

        # k = 0
        # for i in range(truth_patches.shape[0]):
        #     ratio = np.sum(truth_patches[i, :, :]) / (patch_size ** 2)
        #     if ratio <0.1:
        #         k+=1

        '''write image patches into jpg/png files'''
        patch_dir = os.path.join(imageDir, '{}_{:0>2}_patches_size{}_balanced'.format(city_name, j, patch_size))
        if not os.path.isdir(patch_dir):
            os.makedirs(patch_dir)
        else:
            print("Cleaning old data")
            os.chdir(patch_dir)
            filelist = glob.glob("*.png")
            for f in filelist:
                os.remove(f)

        print("Saving patches into {}".format(patch_dir))

        file_path = os.path.expanduser("~/Documents/data/building/patch_list")
        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        with open(os.path.join(file_path, '{}_{:0>2}.txt'.format(city_name, j)), 'w') as file:
            for k in range(image_patches.shape[0]):
                # get image/truth patches
                image_temp = image_patches[k]
                truth_temp = truth_patches[k]
                image_name_temp = os.path.join(patch_dir, "image_patch_{:0>6}.png".format(k))
                truth_name_temp = os.path.join(patch_dir, "truth_patch_{:0>6}.png".format(k))
                file.write(image_name_temp + " " + truth_name_temp + '\n')
                # file.write(patch_dir+image_name_temp+", "+patch_dir+truth_name_temp + '\n')
                misc.imsave(image_name_temp, image_temp)
                misc.imsave(truth_name_temp, truth_temp)
            print("Save {} patches".format(image_patches.shape[0]))
