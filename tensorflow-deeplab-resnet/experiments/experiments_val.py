import subprocess
import os
from time import sleep



NUM_CLASSES = 2

# solar cell data
# learning_rates = [1e-4]
# weight_decay = 0.0005
# batch_sizes = [40]

# IMAGE_PATH = os.path.expanduser("~/Documents/data/igarssTrainingAndTestingData/imageFiles")
# IMAGE_SIZE = 128

# building data
train_city_list = ['Norfolk', 'Arlington', 'Austin', 'Seekonk', 'NewHaven']
test_city_list = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
resolutionList = {'Norfolk': 1.0, 'Arlington': 0.984252, 'Atlanta': 0.5, 'Austin': 0.5, 'Seekonk': 0.984252, 'NewHaven': 0.984252}
# test_city_list = ['NewHaven']
ind = 1
train_city_list = ['Norfolk', 'Arlington', 'Austin', 'NewHaven']

test_city_list = ['Atlanta', 'Austin', 'Seekonk', 'NewHaven']


learning_rates = [1e-3]
batch_sizes = [20]
input_size = '128,128'
decay_step = 10 # in epochs
decay_rate = 0.1
IMAGE_SIZE = 128
weight_decay = 0.0005

GPU = 0
BATCH_SIZE_VAL = 50
for train_city_name in train_city_list:
    for test_city_name in test_city_list:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                # if (batch_size != 20) | (learning_rate != 1e-5) | (weight_decay != 0.0005):
                # snapshot_path = '../snapshots/train_with_pretrained_model/sc_all_batchsize{}_learningRate_{:.0e}_weight_decay_{}'.format(batch_size, learning_rate, weight_decay)
                # snapshot_path = '../snapshots_building/train_with_pretrained_model/batchsize{}_learningRate_{:.0e}_weight_decay_{}'.format(batch_size, learning_rate, weight_decay)
                # if train_city_name == 'Norfolk':
                #     decay_step = 10

                snapshot_path = '../snapshots_building/train_with_pretrained_model/{}_{:0>2}_batchsize{}_learningRate_{:.0e}_L2weight_{}_decayStep_{:d}_decayRate{}/'.format(train_city_name, ind, batch_size, learning_rate, weight_decay, decay_step, decay_rate)

                SAVE_DIR = os.path.join(snapshot_path, 'images_NT')  #

                res_ratio = resolutionList[test_city_name] / resolutionList[train_city_name]
                IMAGE_PATH = os.path.expanduser("~/Documents/data/building/{}".format(test_city_name))

                # if train_city_name == test_city_name:
                if True:
                    subprocess.call(['python', '../validation.py',
                                     '--img_path={}'.format(IMAGE_PATH),
                                     '--restore-from={}'.format(snapshot_path),
                                     '--evaluation-data={}'.format(test_city_name),
                                     '--training-data={}'.format(train_city_name),
                                     # "--is-mean-transfer",
                                     "--resolution-ratio={}".format(res_ratio),
                                     '--num-classes={}'.format(NUM_CLASSES),
                                     "--image-size={}".format(IMAGE_SIZE),
                                     '--batch-size={}'.format(BATCH_SIZE_VAL),
                                     '--save-dir={}'.format(SAVE_DIR),
                                     '--GPU={}'.format(GPU)])

                # subprocess.call(['python', '../validation.py',
                #                  '--img_path={}'.format(IMAGE_PATH),
                #                  '--restore-from={}'.format(snapshot_path),
                #                  '--evaluation-data={}'.format(test_city_name),
                #                  '--training-data={}'.format(train_city_name),
                #                  "--is-mean-transfer",
                #                  "--resolution-ratio={}".format(res_ratio),
                #                  '--num-classes={}'.format(NUM_CLASSES),
                #                  "--image-size={}".format(IMAGE_SIZE),
                #                  '--batch-size={}'.format(BATCH_SIZE_VAL),
                #                  '--save-dir={}'.format(SAVE_DIR),
                #                  '--GPU={}'.format(GPU)])





