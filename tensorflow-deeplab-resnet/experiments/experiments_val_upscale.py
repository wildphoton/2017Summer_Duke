import subprocess
import os
from time import sleep

learning_rates = [1e-4]
weight_decay = 0.0005
batch_sizes = [20]

NUM_CLASSES = 2
IMAGE_PATH = os.path.expanduser("~/Documents/data/igarssTrainingAndTestingData/imageFiles")
IMAGE_SIZE = 41
GPU = 1
BATCH_SIZE_VAL=30
UP_SCALE = 2

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        # if (batch_size != 20) | (learning_rate != 1e-5) | (weight_decay != 0.0005):
        SNAPSHOT_DIR = '../snapshots/train_upscale{}_with_pretrained_model/sc_all_batchsize{}_learningRate_{:.0e}_weight_decay_{}'.format(UP_SCALE, batch_size, learning_rate, weight_decay)
        SAVE_DIR = os.path.join(SNAPSHOT_DIR, 'images')  #
        # for i in range(20):
        subprocess.call(['python', '../validation_upscale.py',
                         '--img_path={}'.format(IMAGE_PATH),
                         '--restore-from={}'.format(SNAPSHOT_DIR),
                         '--num-classes={}'.format(NUM_CLASSES),
                         '--up-scale={}'.format(UP_SCALE),
                         '--batch-size={}'.format(BATCH_SIZE_VAL),
                         '--save-dir={}'.format(SAVE_DIR),
                         '--GPU={}'.format(GPU)])
        # sleep(600)