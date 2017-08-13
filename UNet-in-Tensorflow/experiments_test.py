"""
This script is for running experiments that testing models trained on building data.
Test images are large images that are divided into small patches.
Then the testing results of them are stitched into the orginal size to evaluate performance
"""
import subprocess
import os

# building data
# city list used for defining trained model. Choose different list depends on your experiments
train_city_list = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
train_city_list = ['all']  # the all model does not include Seekonk
train_city_list = ["all_but_{}".format(city_name) for city_name in ['Atlanta', 'Austin', 'NewHaven']]  # all but one city list (Seekonk is always excluded)

# cities of which images are tested
test_city_list = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']

# parameters for defined the trained model
learning_rates = [1e-3]
batch_sizes = [20]
IMAGE_SIZE = 128
decay_step = 1 # in epochs
decay_rate = 0.9

# parameter for testing
GPU = 1  # which GPU to use
BATCH_SIZE_VAL = 200  # testing batch size
THRESHOLD = 0.5 # threshold from confidence map to binary prediction result

for train_city_name in train_city_list:
    ind_training = 1
    for test_city_name in test_city_list:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                snapshot_path = './snapshots_building_balanced/{}_{:0>2}_loss_entropy_batchsize{}_learningRate_{:.0e}_decayStep_{}_decayRate{}'.format(train_city_name, ind_training, batch_size, learning_rate, decay_step, decay_rate)

                # model trained with balanced data
                # snapshot_path = './snapshots_building/{}_{:0>2}_loss_entropy_batchsize{}_learningRate_{:.0e}_decayStep_{}_decayRate{}'.format(train_city_name, ind_training, batch_size, learning_rate, decay_step, decay_rate)

                # model trained with balanced data with IOU as lost function
                # snapshot_path = './snapshots_building/{}_{:0>2}_loss_IOU_batchsize{}_learningRate_{:.0e}_decayStep_{}_decayRate{}'.format(train_city_name, ind_training, batch_size, learning_rate, decay_step, decay_rate)

                SAVE_DIR = os.path.join(snapshot_path, 'images')  # where to save the test result images

                res_ratio = 1.0 # change this if the test image has different resolution with training images, and it will adapt the test image to the resolution of training images

                IMAGE_PATH = os.path.expanduser("~/Documents/data/building/")

                if True:
                    subprocess.call(['python', './test.py',
                                     '--img_path={}'.format(IMAGE_PATH),
                                     '--restore-from={}'.format(snapshot_path),
                                     '--testing-data={}'.format(test_city_name),
                                     '--training-data={}'.format(train_city_name),
                                     "--is-mean-transfer",  # if to use the mean of test image
                                     # "--is-CORAL", # if use CORAL to adapt test image to training image
                                     "--resolution-ratio={}".format(res_ratio),
                                     "--image-size={}".format(IMAGE_SIZE),
                                     '--batch-size={}'.format(BATCH_SIZE_VAL),
                                     '--save-dir={}'.format(SAVE_DIR),
                                     "--pred-threshold={}".format(THRESHOLD),
                                     '--GPU={}'.format(GPU)])

                # you can use the second testing way for the same model-testing_data pair
                # if True:
                #     subprocess.call(['python', './test.py',
                #                      '--img_path={}'.format(IMAGE_PATH),
                #                      '--restore-from={}'.format(snapshot_path),
                #                      '--testing-data={}'.format(test_city_name),
                #                      '--training-data={}'.format(train_city_name),
                #                      "--is-mean-transfer",  # if to use the mean of test image
                #                      # "--is-CORAL", # if use CORAL to adapt test image to training image
                #                      "--resolution-ratio={}".format(res_ratio),
                #                      "--image-size={}".format(IMAGE_SIZE),
                #                      '--batch-size={}'.format(BATCH_SIZE_VAL),
                #                      '--save-dir={}'.format(SAVE_DIR),
                #                      "--pred-threshold={}".format(THRESHOLD),
                #                      '--GPU={}'.format(GPU)])

