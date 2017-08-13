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
IMG_MEAN = np.array((121.68045527, 132.14961763, 129.30317439), dtype=np.float32) # mean of solar panel data in BGR order

# parameters for network
NUM_CLASSES = 2
IMAGE_PATH = os.path.expanduser("~/Documents/data/igarssTrainingAndTestingData/imageFiles")
batch_size = 16
learning_rate = 1e-3
SNAPSHOT_DIR = './snapshots/sc_batchsize{}_learningRate_{:.0e}'.format(batch_size, learning_rate)
SAVE_DIR = os.path.join(SNAPSHOT_DIR, 'images')  # './output/'
IMAGE_SIZE = 41
GPU = '0'
UP_SCALE = 2;
# parameters for validation
valid_stride = 20
valid_batch_size = 50
valid_list = ['11ska595800{}', '11ska460755{}', '11ska580860{}', '11ska565845{}']


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
    parser.add_argument("--up-scale", type=int, default=UP_SCALE,
                        help="the scale that upsample the input image")
    parser.add_argument("--batch-size", type=int, default=valid_batch_size,
                        help="Number of validation image patches sent to the network in one step.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--GPU", type=str, default=GPU,
                        help="GPU used for computation")

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

    # setup used GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    """Create the model and start the evaluation process."""

    # data reader.

    # input image
    input_img = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    # img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=3, num_or_size_splits=3, value=input_img)
    img = tf.cast(tf.concat(axis=3, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    img_upscale = tf.image.resize_bilinear(img, [IMAGE_SIZE*args.up_scale, IMAGE_SIZE*args.up_scale])

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
            print("Model restored from {}".format(ckpt.model_checkpoint_path))
        else:
            print("No model found at{}".format(args.restore_from))
    elif os.path.isfile(args.restore_from):
        # load checkpoint file
        load(loader, sess, args.restore_from)
    else:
        print("No model found at{}".format(args.restore_from))

    '''Perform validation on large images.'''
    # preds, scoremap, pmap, cnn_out, fc0, fc1, fc2, fc3 = sess.run([pred, raw_output, raw_output_up, res5c_relu, fc1_voc12_c0, fc1_voc12_c1, fc1_voc12_c2, fc1_voc12_c3], feed_dict={input_img})

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # gaussian weight kernel
    gfilter = gauss2D(shape=[IMAGE_SIZE, IMAGE_SIZE], sigma=(IMAGE_SIZE - 1) / 4)

    seg_metric = SegMetric(1)

    for valid_file in valid_list:
        print("Validate image {}".format(valid_file[0:-2]))
        valid_image = misc.imread(os.path.join(args.img_path, valid_file.format('.png')))
        valid_truth = (misc.imread(os.path.join(args.img_path, valid_file.format('_truth.png')))/255).astype(np.uint8)
        image_shape = valid_truth.shape

        valid_patches = patchify(valid_image, IMAGE_SIZE, valid_stride)

        """divided patches into smaller batch for validation"""
        pred_pmap = valid_in_batch(valid_patches, sess, pmap, input_img, step=valid_batch_size)

        # pred_pmap = np.ones(valid_patches.shape[0:-1])

        print("Stiching patches")
        pred_pmap_weighted = pred_pmap * gfilter[None, :, :]
        pred_pmap_weighted_large = unpatchify(pred_pmap_weighted, image_shape, valid_stride)
        gauss_mask_large = unpatchify(np.ones(pred_pmap.shape) * gfilter[None, :, :], image_shape, valid_stride)
        pred_pmap_weighted_large_normalized = np.nan_to_num(pred_pmap_weighted_large / gauss_mask_large)
        pred_binary = (pred_pmap_weighted_large_normalized > 0.5).astype(np.uint8)

        # mean IoU
        seg_metric.add_image_pair(pred_binary, valid_truth)
        print("mean_IU: {:.4f}".format(mean_IU(pred_binary, valid_truth)))


        # print("Save validation prediction")
        misc.imsave(os.path.join(args.save_dir, '{}_valid_pred.png'.format(valid_file[0:-2])), pred_binary)
        misc.imsave(os.path.join(args.save_dir, '{}_valid_pred_255.png'.format(valid_file[0:-2])), pred_binary*255)
        misc.toimage(pred_pmap_weighted_large_normalized.astype(np.float32), high=1.0, low=0.0, cmin=0.0, cmax=1.0, mode='F').save(
            os.path.join(args.save_dir, '{}_valid_pmap.tif'.format(valid_file[0:-2])))


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

    print("Overal mean IoU: {:.4f}".format(seg_metric.mean_IU()))
    print('The output file has been saved to {}'.format(args.save_dir))


# plt.show()

if __name__ == '__main__':
    main()
