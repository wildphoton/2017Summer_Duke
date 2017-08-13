import subprocess

DATA_DIRECTORY = '/home/helios/Documents/data/igarssTrainingAndTestingData/training_patches_images_size41_all'
DATA_LIST_PATH = '../dataset/train_sc_all.txt'
INITIAL_MODEL = '/home/helios/Documents/data/Model_zoo/deeplab_resnet.ckpt'

up_scale = 2
learning_rates = [1e-4]
batch_sizes = [20]
num_epochs = 1
GPU = 1
weight_decay = 0.0005

for learning_rate in learning_rates:
    for batch_size in batch_sizes:

        snapshot_path = '../snapshots/train_upscale{}_with_pretrained_model/sc_all_batchsize{}_learningRate_{:.0e}_weight_decay_{}/'\
            .format(up_scale, batch_size, learning_rate,  weight_decay)

        subprocess.call(['python', '../train_upscale.py',
                         '--not-restore-last',
                         '--batch-size={}'.format(batch_size),
                         "--up-scale={}".format(up_scale),
                         '--learning-rate={:.0e}'.format(learning_rate),
                         '--num-steps={}'.format(int(num_epochs * 2223366 / batch_size)),
                         '--weight-decay={}'.format(weight_decay),
                         "--data-dir={}".format(DATA_DIRECTORY),
                         "--data-list={}".format(DATA_LIST_PATH),
                         "--save-pred-every={}".format(1000),
                         "--save-num-images={}".format(max(10, batch_size)),
                         "--restore-from={}".format(INITIAL_MODEL),
                         '--snapshot-dir={}'.format(snapshot_path),
                         '--GPU={}'.format(GPU)])