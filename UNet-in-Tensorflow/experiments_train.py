"""
This script is for running experiments that training models on building data.

"""
import subprocess
import os

# training hyperparameters
learning_rates = [1e-3]
batch_sizes = [20]
num_epochs = 30
input_size = '128,128'
decay_step = 1 # in epochs
decay_rate = 0.9
# decay_rate = 0.95

# city list used for training models. Choose different list depends on your experiments
citylist = ['Norfolk', 'Arlington', 'Atlanta', 'Seekonk', 'NewHaven', 'Austin']  # individual city list
citylist = ['all']  # the all model does not include Seekonk
citylist = ["all_but_{}".format(city_name) for city_name in ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'NewHaven']] # all but one city list (Seekonk is always excluded)
NUM_CLASSES = 2

GPU = 0 # set GPU index

for city_name in citylist:

    # the file of training patch file list
    ind_training = 1
    TRAINING_DATA_LIST_PATH = os.path.expanduser('~/Documents/data/building/patch_list/{}_{:0>2}.txt'.format(city_name, ind_training))

    # the file of validation patch file list
    ind_validation = 2
    VALID_DATA_LIST_PATH = os.path.expanduser('~/Documents/data/building/patch_list/{}_{:0>2}.txt'.format(city_name, ind_validation))


    for batch_size in batch_sizes:
        for learning_rate in learning_rates:

            # model trained with balanced data
            snapshot_path = './snapshots_building_balanced/{}_{:0>2}_loss_entropy_batchsize{}_learningRate_{:.0e}_decayStep_{}_decayRate{}'.format(city_name, ind_training, batch_size, learning_rate, decay_step, decay_rate)

            # model trained with unbalanced data
            # snapshot_path = './snapshots_building/{}_{:0>2}_loss_entropy_batchsize{}_learningRate_{:.0e}_decayStep_{}_decayRate{}'.format(city_name, ind_training, batch_size, learning_rate, decay_step, decay_rate)

            # models trained with unbalanced data with IOU loss
            # snapshot_path = './snapshots_building/{}_{:0>2}_loss_IOU_batchsize{}_learningRate_{:.0e}_decayStep_{}_decayRate{}'.format(city_name, ind_training, batch_size, learning_rate, decay_step, decay_rate)

            # call train.py for training, see its documentation for details of args
            subprocess.call(['python', './train.py',
                             '--training-data={}'.format(city_name),
                             '--num-classes={}'.format(NUM_CLASSES),
                             '--batch-size={}'.format(batch_size),
                             '--learning-rate={:.0e}'.format(learning_rate),
                             "--epochs={:d}".format(num_epochs),
                             "--decay-step={}".format(decay_step),
                             '--decay-rate={}'.format(decay_rate),
                             '--training-data-list={}'.format(TRAINING_DATA_LIST_PATH),
                             '--validation-data-list={}'.format(VALID_DATA_LIST_PATH),
                             "--input-size={}".format(input_size),
                             "--is-loss-entropy", # comment this arg to use IOU as loss function
                             '--random-mirror',
                             '--random-scale',
                             "--random-rotate",
                             '--ckdir={}'.format(snapshot_path),
                             '--GPU={}'.format(GPU)])
