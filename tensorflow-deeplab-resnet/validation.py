"""Run DeepLab-ResNet on large images.

This script computes a probability map for a given image,
computer PR curve based on the probablity map and ground truth,
compute mean IOU from the solar panel mask by thresholding the pmap.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc as misc

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

sys.path.append(os.path.realpath('../metrics'))
sys.path.append(os.path.realpath('../dataReader'))
sys.path.append(os.path.realpath('../utils'))

from seg_metric import SegMetric
from patchify import patchify, unpatchify, gauss2D
from eval_segm import *
from seg_metric import SegMetric

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) # mean for raw ResNet

# parameters for network
NUM_CLASSES = 2
IMAGE_PATH = os.path.expanduser("~/Documents/data/igarssTrainingAndTestingData/imageFiles")
batch_size = 16
learning_rate = 1e-3
SNAPSHOT_DIR = './snapshots/sc_batchsize{}_learningRate_{:.0e}'.format(batch_size, learning_rate)
SAVE_DIR = os.path.join(SNAPSHOT_DIR, 'images')  # './output/'
IMAGE_SIZE = 41
GPU = '0'

train_city_name = 'Norfolk'
valid_city_name = 'Austin'
ind = 1

IMAGE_PATH = os.path.expanduser("~/Documents/data/building/{}".format(valid_city_name))
data_option = 'building_{}'.format(valid_city_name)
learning_rates = 1e-3
batch_size = 20
num_epochs = 45
input_size = '128,128'
decay_step = 15 # in epochs
decay_rate = 0.1
IMAGE_SIZE = 128
weight_decay = 0.0005
SNAPSHOT_DIR = '../snapshots_building/train_with_pretrained_model/{}_{:0>2}_batchsize{}_learningRate_{:.0e}_L2weight_{}_decayStep_{:d}_decayRate{}/'.format(
    train_city_name, ind, batch_size, learning_rate, weight_decay, decay_step, decay_rate)
SAVE_DIR = os.path.join(SNAPSHOT_DIR, 'images')  # './output/'

valid_batch_size = 50

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--img_path", type=str, default=IMAGE_PATH,
                        help="Path to the RGB image file.")
    parser.add_argument("--restore-from", type=str, default=SNAPSHOT_DIR,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                        help="Size of image (by default h=w)")
    parser.add_argument("--batch-size", type=int, default=valid_batch_size,
                        help="Number of evaluation image patches sent to the network in one step.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--GPU", type=str, default=GPU,
                        help="GPU used for computation")
    parser.add_argument("--evaluation-data", type=str, default=valid_city_name, help="which data used for evaluation: SP - solar panel data; $cityname$ - building data at $cityname$")
    parser.add_argument("--training-data", type=str, default=train_city_name, help="which data used for training: SP - solar panel data; $cityname$ - building data at $cityname$")
    parser.add_argument("--is-mean-transfer", action="store_true",
                        help="Whether to to use the image mean from testing data.")
    parser.add_argument("--resolution-ratio", type = float,
                        default = 1.0, help = "the ration of resolution between evaluation data and training data")
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def valid_in_batch(patches, session, out_var, input_var, step=100):
    num_patches = patches.shape[0]
    print("Testing total {} patch".format(num_patches))
    output = []
    for itr in range(0, int(np.ceil(num_patches / float(step)))):
        temp_output = session.run(out_var, feed_dict={
            input_var: patches[itr * step:min((itr + 1) * step, num_patches), :, :, :]})
        if itr == 0:
            output = temp_output[:, :, :, 1]
        else:
            output = np.concatenate((output, temp_output[:, :, :, 1]), axis=0)
        # print("Tested {} to {} patches".format(itr * step, min((itr + 1) * step, num_patches)))

    return output


def main():
    # get arguments
    args = get_arguments()

    IMG_MEAN = np.zeros(3)
    valid_list=[]

    # parameters of building data set
    citylist = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
    image_mean_list = {'Norfolk': [127.07435926, 129.40160709, 128.28713284],
                       'Arlington': [88.30304996, 94.97338776, 93.21268212],
                       'Atlanta': [101.997014375, 108.42171833, 110.044871],
                       'Austin': [97.0896012682, 102.94697026, 100.7540157],
                       'Seekonk': [86.67800904, 93.31221168, 92.1328146],
                       'NewHaven': [106.7092798, 111.4314, 110.74903832]} # BGR mean for the training data for each city
    num_samples = {'Norfolk': 3,
                      'Arlington': 3,
                      'Atlanta': 3,
                      'Austin': 3,
                      'Seekonk': 3,
                      'NewHaven': 2} # number of samples for each city
    # set evaluation data
    if args.evaluation_data == 'SP':
        IMG_MEAN = np.array((121.68045527, 132.14961763, 129.30317439),
                        dtype=np.float32)  # mean of solar panel data in BGR order
        IMG_MEAN = [IMG_MEAN[2], IMG_MEAN[1], IMG_MEAN[0]] # convert to RGB order

        # valid_list = [ '11ska505665{}', '11ska580710{}', '11ska475635{}', '11ska475875{}', '11ska565905{}', '11ska490860{}', '11ska325740{}', '11ska460725{}', '11ska490605{}', '11ska430815{}', '11ska400740{}', '11ska580875{}', '11ska655725{}', '11ska595860{}', '11ska460890{}', '11ska655695{}', '11ska640605{}', '11ska580605{}', '11ska595665{}', '11ska505755{}', '11ska475650{}', '11ska595755{}', '11ska625755{}', '11ska490740{}', '11ska565755{}', '11ska520725{}', '11ska595785{}', '11ska580755{}', '11ska445785{}', '11ska625710{}', '11ska520830{}', '11ska640800{}', '11ska535785{}', '11ska430905{}', '11ska505695{}', '11ska565770{}']
        # valid_list = ['11ska580860{}', '11ska565845{}']
        valid_list = ['11ska625680{}', '11ska610860{}', '11ska445890{}', '11ska520695{}', '11ska355800{}', '11ska370755{}',
                  '11ska385710{}', '11ska550770{}', '11ska505740{}', '11ska385800{}', '11ska655770{}', '11ska385770{}',
                  '11ska610740{}', '11ska550830{}', '11ska625830{}', '11ska535740{}', '11ska520815{}', '11ska595650{}',
                  '11ska475665{}', '11ska520845{}']

    elif args.training_data in citylist:
        IMG_MEAN = image_mean_list[args.training_data]
        IMG_MEAN = [IMG_MEAN[2], IMG_MEAN[1], IMG_MEAN[0]] # convert to RGB order
        valid_list = ["{}_{:0>2}{{}}".format(args.evaluation_data, i) for i in range(1,num_samples[args.evaluation_data]+1)]

    else:
        print("Wrong data option: {}".format(args.training_data))

    # set image mean

    # setup used GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    """Create the model and start the evaluation process."""

    # data reader.

    # input image
    input_img = tf.placeholder(tf.float32, shape=[None, args.image_size, args.image_size, 3], name="input_image")
    # img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=3, num_or_size_splits=3, value=input_img)
    img = tf.cast(tf.concat(axis=3, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.

    # Create network.
    net = DeepLabResNetModel({'data': img}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    res5c_relu = net.layers['res5c_relu']
    fc1_voc12_c0 = net.layers['fc1_voc12_c0']
    fc1_voc12_c1 = net.layers['fc1_voc12_c1']
    fc1_voc12_c2 = net.layers['fc1_voc12_c2']
    fc1_voc12_c3 = net.layers['fc1_voc12_c3']

    raw_output = net.layers['fc1_voc12']

    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[1:3, ])
    # raw_output_up_argmax = tf.argmax(raw_output_up, dimension=3)
    # pred = tf.expand_dims(raw_output_up_argmax, dim=3)
    pmap = tf.nn.softmax(raw_output_up, name="probability_map")

    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)

    if os.path.isdir(args.restore_from):
        # search checkpoint at given path
        ckpt = tf.train.get_checkpoint_state(args.restore_from)
        if ckpt and ckpt.model_checkpoint_path:
            # load checkpoint file
            load(loader, sess, ckpt.model_checkpoint_path)
            file = open(os.path.join(args.restore_from, 'test.csv'), 'a')
            file.write("\nTest Model: {}\ntransfer_mean:{}\n".format(ckpt.model_checkpoint_path, args.is_mean_transfer))
        else:
            print("No model found at{}".format(args.restore_from))
            sys.exit()
    elif os.path.isfile(args.restore_from):
        # load checkpoint file
        load(loader, sess, args.restore_from)
        file = open(os.path.join(args.restore_from, 'test.csv'), 'a')
        file.write("\nTest Model: {}\ntransfer_mean:{}\n".format(args.restore_from, args.is_mean_transfer))
    else:
        print("No model found at{}".format(args.restore_from))
        sys.exit()

    '''Perform evaluation on large images.'''
    # preds, scoremap, pmap, cnn_out, fc0, fc1, fc2, fc3 = sess.run([pred, raw_output, raw_output_up, res5c_relu, fc1_voc12_c0, fc1_voc12_c1, fc1_voc12_c2, fc1_voc12_c3], feed_dict={input_img})

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # gaussian weight kernel
    gfilter = gauss2D(shape=[args.image_size, args.image_size], sigma=(args.image_size - 1) / 4)

    seg_metric = SegMetric(1)

    valid_stride = int(args.image_size/2)


    for valid_file in valid_list:
        print("evaluate image {}".format(valid_file[0:-2]))
        if args.evaluation_data == 'SP':
            valid_image = misc.imread(os.path.join(args.img_path, valid_file.format('.png')))
        else:
            valid_image = misc.imread(os.path.join(args.img_path, valid_file.format('_RGB.png')))
        valid_truth = (misc.imread(os.path.join(args.img_path, valid_file.format('_truth.png')))/255).astype(np.uint8)

        valid_image = misc.imresize(valid_image, args.resolution_ratio, interp='bilinear')
        valid_truth = misc.imresize(valid_truth, args.resolution_ratio, interp='nearest')

        if args.is_mean_transfer:
            IMG_MEAN = np.mean(valid_image, axis=(0,1)) # Image mean of testing data

        valid_image = valid_image - IMG_MEAN # substract mean from image

        image_shape = valid_truth.shape

        valid_patches = patchify(valid_image, args.image_size, valid_stride)
        """divided patches into smaller batch for evaluation"""
        pred_pmap = valid_in_batch(valid_patches, sess, pmap, input_img, step=args.batch_size)

        # pred_pmap = np.ones(valid_patches.shape[0:-1])

        print("Stiching patches")
        pred_pmap_weighted = pred_pmap * gfilter[None, :, :]
        pred_pmap_weighted_large = unpatchify(pred_pmap_weighted, image_shape, valid_stride)
        gauss_mask_large = unpatchify(np.ones(pred_pmap.shape) * gfilter[None, :, :], image_shape, valid_stride)
        pred_pmap_weighted_large_normalized = np.nan_to_num(pred_pmap_weighted_large / gauss_mask_large)
        pred_binary = (pred_pmap_weighted_large_normalized > 0.5).astype(np.uint8)
        
        print("Save evaluation prediction")

        misc.imsave(os.path.join(args.save_dir, '{}_valid_pred.png'.format(valid_file[0:-2])), pred_binary)
        misc.imsave(os.path.join(args.save_dir, '{}_valid_pred_255.png'.format(valid_file[0:-2])), pred_binary*255)
        misc.toimage(pred_pmap_weighted_large_normalized.astype(np.float32), high=1.0, low=0.0, cmin=0.0, cmax=1.0, mode='F').save(
            os.path.join(args.save_dir, '{}_valid_pmap.tif'.format(valid_file[0:-2])))

        # mean IoU
        seg_metric.add_image_pair(pred_binary, valid_truth)
        message_temp = "{}, {:.4f}".format(valid_file[0:-2], mean_IU(pred_binary, valid_truth))
        print(message_temp)
        file.write(message_temp+'\n')
        # # Plot PR curve
        # precision, recall, thresholds = precision_recall_curve(valid_truth.flatten(), pred_pmap_weighted_large_normalized.flatten(), 1)
        # plt.figure()
        # plt.plot(recall, precision, lw=2, color='navy',
        #          label='Precision-Recall curve')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('Precision-Recal')
        # # plt.legend(loc="lower left")
        # plt.savefig(os.path.join(args.save_dir, '{}_PR_curve.png'.format(valid_file[0:-2])))

    # msk = decode_labels(preds, num_classes=args.num_classes)
    # im = Image.fromarray(msk[0])

    # im.save(args.save_dir + 'pred.png')
    message_overall = "Overall, {:.4f}".format(seg_metric.mean_IU())
    print(message_overall)
    file.write(message_overall + '\n')
    file.close()
    print('The output file has been saved to {}'.format(args.save_dir))

    sess.close()


# plt.show()

if __name__ == '__main__':
    main()
