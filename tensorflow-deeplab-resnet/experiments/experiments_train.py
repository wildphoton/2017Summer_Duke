import subprocess

# DATA_DIRECTORY = '/home/helios/Documents/data/igarssTrainingAndTestingData/training_patches_images_size41_all'
# DATA_LIST_PATH = '../dataset/train_sc_all.txt'
# data_size = 2223366
# input_size = '41,41'



INITIAL_MODEL = '/home/helios/Documents/data/Model_zoo/deeplab_resnet.ckpt'

learning_rates = [1e-3]
batch_sizes = [20]
input_size = '128,128'
decay_step = 10 # in epochs
decay_rate = 0.1

num_epochs = 30
# citylist = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
citylist = ['Austin']

data_size_list = {'Norfolk': 23409,
               'Arlington': 23409,
               'Atlanta': 22448,
               'Austin': 112367,
               'Seekonk': 23409,
               'NewHaven': 23409}  # number of samples for each city
GPU = 1
weight_decays = [0.0005]
for city_name in citylist:
    ind = 1
    DATA_DIRECTORY = '/home/helios/Documents/data/building/{}/{}_{:0>2}_patches_size128'.format(city_name, city_name, ind)
    DATA_LIST_PATH = '../dataset_building/{}_{:0>2}.txt'.format(city_name, ind)
    data_size = data_size_list[city_name]
    for weight_decay in weight_decays:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                # snapshot_path = '../snapshots/train_with_pretrained_model/sc_all_batchsize{}_learningRate_{:.0e}_weight_decay_{}/'.format(batch_size, learning_rate,  weight_decay)
                snapshot_path = '../snapshots_building/train_with_pretrained_model/{}_{:0>2}_batchsize{}_learningRate_{:.0e}_L2weight_{}_decayStep_{}_decayRate{}/' \
                    .format(city_name, ind, batch_size, learning_rate, weight_decay, decay_step, decay_rate)

                subprocess.call(['python', '../train.py',
                                 '--not-restore-last',
                                 '--training-data={}'.format(city_name),
                                 '--batch-size={}'.format(batch_size),
                                 '--learning-rate={:.0e}'.format(learning_rate),
                                 "--num-epochs={:d}".format(num_epochs),
                                 "--data-size={:d}".format(data_size),
                                 "--decay-step={}".format(decay_step),
                                 '--decay-rate={}'.format(decay_rate),
                                 '--weight-decay={}'.format(weight_decay),
                                 '--data-dir={}'.format(DATA_DIRECTORY),
                                 '--data-list={}'.format(DATA_LIST_PATH),
                                 "--input-size={}".format(input_size),
                                 '--random-mirror',
                                 '--random-scale',
                                 "--random-rotate",
                                 '--is-training',
                                 # "--unit-std",
                                 '--save-pred-every={}'.format(1000),
                                 '--restore-from={}'.format(INITIAL_MODEL),
                                 '--snapshot-dir={}'.format(snapshot_path),
                                 '--GPU={}'.format(GPU)])