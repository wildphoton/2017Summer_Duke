import subprocess

DATA_DIRECTORY = '/home/helios/Documents/data/igarssTrainingAndTestingData/training_patches_images_size41_all'
DATA_LIST_PATH = '../dataset/train_sc_all.txt'
INITIAL_MODEL = '/home/helios/Documents/data/Model_zoo/deeplab_resnet.ckpt'
INPUT_SIZE = '41,41'

learning_rates = [1e-4]
batch_sizes = [40]
num_epochs = 1
GPU = 1

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        snapshot_path = '../snapshots/finetune_with_pretrained_model/sc_all_batchsize{}_learningRate_{:.0e}/'\
            .format(batch_size, learning_rate)

        num_steps = int(num_epochs * 2223366 / batch_size)

        subprocess.call(['python', '../fine_tune.py',
                         '--batch-size={}'.format(batch_size),
                         "--data-dir={}".format(DATA_DIRECTORY),
                         "--data-list={}".format(DATA_LIST_PATH),
                         "--input-size={}".format(INPUT_SIZE),
                         '--learning-rate={:.0e}'.format(learning_rate),
                         '--not-restore-last',
                         "--num-classes={}".format(2),
                         '--num-steps={}'.format(num_steps),
                         "--restore-from={}".format(INITIAL_MODEL),
                         "--save-num-images={}".format(10),
                         "--save-pred-every={}".format(1000),
                         '--snapshot-dir={}'.format(snapshot_path),
                         '--GPU={}'.format(GPU)])