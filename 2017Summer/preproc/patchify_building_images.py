import scipy.misc as misc
import os
import numpy as np
import sys
import glob

sys.path.append(os.path.realpath('../dataReader'))
from patchify import patchify
citylist = [ 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
resolutionList = {'Norfolk': 1, 'Arlington': 0.984252, 'Atlanta': 0.5, 'Austin': 0.5, 'Seekonk': 0.984252, 'NewHaven': 0.984252}
dataDir = os.path.expanduser("~/Documents/data/building")

image_mean_list = {'Norfolk': [127.07435926, 129.40160709, 128.28713284],
                   'Arlington': [88.30304996, 94.97338776, 93.21268212],
                   'Atlanta': [101.997014375, 108.42171833, 110.044871],
                   'Austin': [97.0896012682, 102.94697026, 100.7540157],
                   'Seekonk': [86.67800904, 93.31221168, 92.1328146],
                   'NewHaven': [106.7092798, 111.4314, 110.74903832]}
image_std_list = {'Norfolk': [28.615469420031832, 32.662536832452886, 37.64149854207523],
                   'Arlington':[30.40903039206398, 37.973725024862595, 43.58402191634698],
                   'Atlanta':[36.93662467838125, 39.43470059838385, 41.74732676809388],
                   'Austin': [42.81337177109884, 43.71071321350751, 44.440517007230675],
                   'Seekonk':[25.506449467410715, 32.46885262572024, 37.76814267502958],
                   'NewHaven': [33.05784541012469, 36.62685162291547, 37.686084270914435]}

dataDir = os.path.expanduser("~/Documents/data/building")

for i in range(0,6):
    city_name = citylist[i]
    imageDir = os.path.join(dataDir, '{}'.format(city_name))
    num_image = glob.glob(os.path.join(imageDir, "{}_*_RGB.png".format(city_name))).__len__()
    patch_size = 128
    patch_stride = int(patch_size)
    
    for j in range(2,3):
        imageFile = os.path.join(imageDir, "{}_{:0>2}_RGB.png".format(city_name, j))
        truthFile = os.path.join(imageDir, "{}_{:0>2}_truth.png".format(city_name, j))

        image_large = misc.imread(imageFile)
        truth_large = (misc.imread(truthFile)/ 255).astype(np.uint8)

        image_mean = np.mean(image_large, axis=(0,1))
        # print("Image mean of {} in BGR order: [{}, {}, {}]".format(imageFile, image_mean[2], image_mean[1], image_mean[0]))
        image_std = np.std(image_large, axis=(0,1))
        print("Image std of {} in BGR order: [{:3}, {}, {}]".format(imageFile, image_std[2], image_std[1], image_std[0]))
        print("Generating patches")
        image_patches = patchify(image_large, patch_size, patch_stride)
        truth_patches = patchify(truth_large, patch_size, patch_stride)

        '''write image patches into jpg/png files'''
        patch_dir = os.path.join(imageDir, '{}_{:0>2}_patches_size{}'.format(city_name, j, patch_size))
        if not os.path.isdir(patch_dir):
            os.makedirs(patch_dir)

        print("Saving patches into {}".format(patch_dir))

        file_path = os.path.expanduser("~/Documents/zhenlinx/code/tensorflow-deeplab-resnet/dataset_building")
        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        with open(os.path.join(file_path, '{}_{:0>2}.txt'.format(city_name,j)),'w') as file:
            for k in range(image_patches.shape[0]):
                # get image/truth patches
                image_temp = image_patches[k]
                truth_temp = truth_patches[k]
                image_name_temp = "/image_{:0>6}.png".format(k)
                truth_name_temp = "/truth_{:0>6}.png".format(k)
                file.write(image_name_temp + " " + truth_name_temp + '\n')
                # file.write(patch_dir+image_name_temp+", "+patch_dir+truth_name_temp + '\n')
                misc.imsave(patch_dir+image_name_temp, image_temp)
                misc.imsave(patch_dir+truth_name_temp, truth_temp)
        pass
