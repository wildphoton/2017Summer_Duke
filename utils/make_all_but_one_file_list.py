"""
This script makes file list of patches from all but one cities
Files lists of individual cities have been generated
"""
import numpy as np
import os

citylist = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'NewHaven']

file_path = os.path.expanduser("~/Documents/data/building/patch_list")
for j in range(1, 3):
    if j == 1:  # use the first large image as training data
        num_patch = 20000  # number of patches of training set
    elif j == 2:
        num_patch = 2000  # number of patches to validation set

    file_list = {}

    for input_city_name in citylist:
        with open(os.path.join(file_path, '{}_{:0>2}.txt'.format(input_city_name, j)), 'r') as file:
            file_list[input_city_name] = file.readlines()
            np.random.shuffle(file_list[input_city_name])

    for output_city_name in citylist:
        merged_list_temp = []
        for city_name_temp in citylist:
            if city_name_temp != output_city_name:
                merged_list_temp.extend(file_list[city_name_temp][0:int(num_patch / 4)])

        with open(os.path.join(file_path, 'all_but_{}_{:0>2}.txt'.format(output_city_name, j)), 'w') as file:
            file.writelines(merged_list_temp)

    merged_list_all = []
    for city_name_temp in citylist:
        merged_list_all.extend(file_list[city_name_temp][0:int(num_patch / 5)])
    with open(os.path.join(file_path, 'all_{:0>2}.txt'.format(j)), 'w') as file:
        file.writelines(merged_list_all)
