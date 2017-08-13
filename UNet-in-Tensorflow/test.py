"""
Simple U-Net implementation in TensorFlow

Objective: detect vehicles

y = f(X)

X: image (640, 960, 3)
y: mask (640, 960, 1)
   - binary image
   - background is masked 0
   - vehicle is masked 255

Loss function: maximize IOU

    (intersection of prediction & grount truth)
    -------------------------------
    (union of prediction & ground truth)

Notes:
    In the paper, the pixel-wise softmax was used.
    But, I used the IOU because the datasets I used are
    not labeled for segmentations

Original Paper:
    https://arxiv.org/abs/1505.04597
"""
import time
import os
import sys
import pandas as pd
import tensorflow as tf
import scipy.misc as misc

sys.path.append(os.path.realpath('../dataReader'))
from image_reader import ImageReader

sys.path.append(os.path.realpath('../utils'))
from patchify import patchify, unpatchify, gauss2D
from CORAL import image_adapt

sys.path.append(os.path.realpath('../metrics'))
from eval_segm import *
from seg_metric import SegMetric
from seg_metric import SegMetric

from utils import decode_labels, inv_preprocess, prepare_label
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

train_city_name = 'Norfolk'
test_city_name = 'Austin'
ind = 1

NUM_CLASSES = 2

IMAGE_PATH = os.path.expanduser("~/Documents/data/building/{}".format(test_city_name))
data_option = 'building_{}'.format(test_city_name)
learning_rate = 1e-3
batch_size = 20
input_size = '128,128'
decay_step = 15 # in epochs
decay_rate = 0.1
IMAGE_SIZE = 128
weight_decay = 0.0005
SNAPSHOT_DIR = '../snapshots_building/train_with_pretrained_model/{}_{:0>2}_batchsize{}_learningRate_{:.0e}_L2weight_{}_decayStep_{:d}_decayRate{}/'.format(
    train_city_name, ind, batch_size, learning_rate, weight_decay, decay_step, decay_rate)
SAVE_DIR = os.path.join(SNAPSHOT_DIR, 'images')  # './output/'

valid_batch_size = 50
GPU = '0'
def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default=IMAGE_PATH,
                        help="Path to the RGB image file.")
    parser.add_argument("--restore-from", type=str, default=SNAPSHOT_DIR,
                        help="Path to the file with model weights.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                        help="Size of image (by default h=w)")
    parser.add_argument("--batch-size", type=int, default=valid_batch_size,
                        help="Number of evaluation image patches sent to the network in one step.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--GPU", type=str, default=GPU,
                        help="GPU used for computation")
    parser.add_argument("--testing-data", type=str, default=test_city_name,
                        help="which data used for testing: SP - solar panel data; $cityname$ - building data at $cityname$")
    parser.add_argument("--training-data", type=str, default=train_city_name,
                        help="which data used for training: SP - solar panel data; $cityname$ - building data at $cityname$")
    parser.add_argument("--is-mean-transfer", action="store_true",
                        help="Whether to to use the image mean from testing data.")
    parser.add_argument("--is-CORAL", action="store_true",
                        help="Whether to to use CORAL to align the test image data to training data.")
    parser.add_argument("--resolution-ratio", type=float,
                        default=1.0, help="the ration of resolution between evaluation data and training data, use this parameter when the test image have different resolution with training images")
    parser.add_argument("--pred-threshold", type=float, default=0.5,
                        help="the threshold to generate binary predition from confidence map")

    flags = parser.parse_args()
    return flags

# def image_augmentation(image, mask):
#     """Returns (maybe) augmented images
#
#     (1) Random flip (left <--> right)
#     (2) Random flip (up <--> down)
#     (3) Random brightness
#     (4) Random hue
#
#     Args:
#         image (3-D Tensor): Image tensor of (H, W, C)
#         mask (3-D Tensor): Mask image tensor of (H, W, 1)
#
#     Returns:
#         image: Maybe augmented image (same shape as input `image`)
#         mask: Maybe augmented mask (same shape as input `mask`)
#     """
#     concat_image = tf.concat([image, mask], axis=-1)
#
#     maybe_flipped = tf.image.random_flip_left_right(concat_image)
#     maybe_flipped = tf.image.random_flip_up_down(concat_image)
#
#     image = maybe_flipped[:, :, :-1]
#     mask = maybe_flipped[:, :, -1:]
#
#     image = tf.image.random_brightness(image, 0.7)
#     image = tf.image.random_hue(image, 0.3)
#
#     return image, mask
#
#
# def get_image_mask(queue, augmentation=True):
#     """Returns `image` and `mask`
#
#     Input pipeline:
#         Queue -> CSV -> FileRead -> Decode JPEG
#
#     (1) Queue contains a CSV filename
#     (2) Text Reader opens the CSV
#         CSV file contains two columns
#         ["path/to/image.jpg", "path/to/mask.jpg"]
#     (3) File Reader opens both files
#     (4) Decode JPEG to tensors
#
#     Notes:
#         height, width = 640, 960
#
#     Returns
#         image (3-D Tensor): (640, 960, 3)
#         mask (3-D Tensor): (640, 960, 1)
#     """
#     text_reader = tf.TextLineReader(skip_header_lines=0)
#     _, csv_content = text_reader.read(queue)
#
#     image_path, mask_path = tf.decode_csv(csv_content, record_defaults=[[""], [""]])
#
#     image_file = tf.read_file(image_path)
#     mask_file = tf.read_file(mask_path)
#
#     image = tf.image.decode_png(image_file, channels=3)
#     image.set_shape([128, 128, 3])
#     image = tf.cast(image, tf.float32)
#
#     mask = tf.image.decode_png(mask_file, channels=1)
#     mask.set_shape([128, 128, 1])
#     mask = tf.cast(mask, tf.float32)
#     mask = mask / (tf.reduce_max(mask) + 1e-7)
#
#     if augmentation:
#         image, mask = image_augmentation(image, mask)
#
#     return image, mask


def conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool

def upsample_concat(inputA, input_B, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = upsampling_2D(inputA, size=(2, 2), name=name)

    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))

def upsampling_2D(tensor, name, size=(2, 2)):
    """Upsample/Rescale `tensor` by size

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))

    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))

def make_unet(X, training):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """


    # net = X / 127.5 - 1
    # net = tf.layers.conv2d(net, 3, (1, 1), name="color_space_adjust")
    conv1, pool1 = conv_conv_pool(X, [8, 8], training, name='conv1')
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name='conv2')
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name='conv3')
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name='conv4')
    conv5 = conv_conv_pool(pool4, [128, 128], training, name='conv5', pool=False)

    up6 = upsample_concat(conv5, conv4, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, name=6, pool=False)

    up7 = upsample_concat(conv6, conv3, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, name=7, pool=False)

    up8 = upsample_concat(conv7, conv2, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, name=8, pool=False)

    up9 = upsample_concat(conv8, conv1, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, name=9, pool=False)

    return tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=None, padding='same')

def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)

def make_train_op(loss, global_step, learning_rate):
    """Returns a training operation

    Loss function = - IOU(y_pred, y_true)

    IOU is

        (the area of intersection)
        --------------------------
        (the area of two boxes)

    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)

    Returns:
        train_op: minimize operation
    """
    # global_step = tf.train.get_or_create_global_step()

    optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optim.minimize(loss, global_step=global_step)


def myIOU(y_pred, y_true, segmetric):
    """
    iou score on stream
    :param y_pred: 4D array of prediction result
    :param y_true: 4D array of ground truth
    :param segmetric: object of segmentation metric class
    :return: current iou value
    """
    for i in range(y_pred.shape[0]):
        segmetric.add_image_pair(y_pred[i,:,:,0], y_true[i,:,:,0])

    return segmetric.mean_IU()

def valid_in_batch(patches, session, out_var, input_var, step=100):
    num_patches = patches.shape[0]
    print("Testing total {} patch".format(num_patches))
    output = []
    for itr in range(0, int(np.ceil(num_patches / float(step)))):
        temp_output = session.run(out_var, feed_dict={
            input_var: patches[itr * step:min((itr + 1) * step, num_patches), :, :, :]})
        if itr == 0:
            output = temp_output[:, :, :, 0]
        else:
            output = np.concatenate((output, temp_output[:, :, :, 0]), axis=0)
        # print("Tested {} to {} patches".format(itr * step, min((itr + 1) * step, num_patches)))

    return output

def main(flags):
    IMG_MEAN = np.zeros(3)

    # parameters of building data set
    citylist = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
    image_mean_list = {'Norfolk': [127.07435926, 129.40160709, 128.28713284],
                       'Arlington': [88.30304996, 94.97338776, 93.21268212],
                       'Atlanta': [101.997014375, 108.42171833, 110.044871],
                       'Austin': [97.0896012682, 102.94697026, 100.7540157],
                       'Seekonk': [86.67800904, 93.31221168, 92.1328146],
                       'NewHaven': [106.7092798, 111.4314,
                                    110.74903832]}  # BGR mean for the training data for each city

    num_samples = {'Norfolk': 3,
                   'Arlington': 3,
                   'Atlanta': 3,
                   'Austin': 3,
                   'Seekonk': 3,
                   'NewHaven': 2}  # number of samples for each city

    # set evaluation data
    if flags.training_data == 'SP':
        IMG_MEAN = np.array((121.68045527, 132.14961763, 129.30317439),
                            dtype=np.float32)  # mean of solar panel data in BGR order
        valid_list = ['11ska625680{}', '11ska610860{}', '11ska445890{}', '11ska520695{}', '11ska355800{}',
                      '11ska370755{}',
                      '11ska385710{}', '11ska550770{}', '11ska505740{}', '11ska385800{}', '11ska655770{}',
                      '11ska385770{}',
                      '11ska610740{}', '11ska550830{}', '11ska625830{}', '11ska535740{}', '11ska520815{}',
                      '11ska595650{}',
                      '11ska475665{}', '11ska520845{}']

    elif flags.training_data in citylist:
        IMG_MEAN = image_mean_list[flags.training_data] # mean of building data in RGB order
        valid_list = ["{}_{:0>2}{{}}".format(flags.testing_data, i) for i in
                      range(1, num_samples[flags.testing_data] + 1)]


    elif 'all_but' in flags.training_data:
        except_city_name = flags.training_data.split('_')[2]
        for cityname in citylist:
            if cityname != except_city_name and cityname != 'Seekonk':
                IMG_MEAN = IMG_MEAN + np.array(image_mean_list[cityname])
        IMG_MEAN = IMG_MEAN / 4
        valid_list = ["{}_{:0>2}{{}}".format(flags.testing_data, i) for i in
                      range(1, num_samples[flags.testing_data] + 1)]

    elif flags.training_data == 'all':
        for cityname in citylist:
            if cityname != 'Seekonk':
                IMG_MEAN = IMG_MEAN + np.array(image_mean_list[cityname])
        IMG_MEAN = IMG_MEAN / 5
        valid_list = ["{}_{:0>2}{{}}".format(flags.testing_data, i) for i in
                      range(1, num_samples[flags.testing_data] + 1)]

    else:
        print("Wrong data option: {}".format(flags.data_option))

    IMG_MEAN = [IMG_MEAN[2], IMG_MEAN[1], IMG_MEAN[0]]  # convert to RGB order

    # setup used GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.GPU

    # presetting
    tf.set_random_seed(1234)

    # input image batch with zero mean
    image_batch = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name="image_batch")
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=3, num_or_size_splits=3, value=image_batch)
    img_bgr = tf.cast(tf.concat(axis=3, values=[img_b, img_g, img_r]), dtype=tf.float32)

    prediction_batch = tf.placeholder(tf.float32, shape=[None, 128, 128, 1], name="prediction_batch")

    pred_raw = make_unet(img_bgr, training=False)
    pred = tf.nn.sigmoid(pred_raw)
    tf.add_to_collection("inputs", image_batch)
    tf.add_to_collection("outputs", pred)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables())

        if os.path.exists(flags.restore_from) and tf.train.get_checkpoint_state(flags.restore_from):
            latest_check_point = tf.train.latest_checkpoint(flags.restore_from)
            print("Loading model: {}".format(latest_check_point))
            saver.restore(sess, latest_check_point)
        else:
            print("No model found at{}".format(flags.restore_from))
            sys.exit()

        if not os.path.exists(flags.save_dir):
            os.makedirs(flags.save_dir)

        # testing model on large images by running over patches
        gfilter = gauss2D(shape=[flags.image_size, flags.image_size], sigma=(flags.image_size - 1) / 4)
        seg_metric = SegMetric(1)
        valid_stride = int(flags.image_size / 2)

        print("Testing {} model on {} data {}{}".format(flags.training_data, flags.testing_data, "with transferring mean" if flags.is_mean_transfer else "", "with CORAL domain adaption" if flags.is_CORAL else ""))

        file = open(os.path.join(flags.restore_from, 'test_log.csv'), 'a')
        file.write("\nTest Model: {}\ntransfer_mean:{} CORAL domain adaption:{}\n".format(latest_check_point, flags.is_mean_transfer, flags.is_CORAL))

        for valid_file in valid_list:
            print("Testing image {}".format(valid_file[0:-2]))
            if flags.testing_data == 'SP':
                valid_image = misc.imread(os.path.join(flags.img_path, valid_file.format('.png')))
            else:
                valid_image = misc.imread(os.path.join(flags.img_path, flags.testing_data, valid_file.format('_RGB_1feet.png')))

            valid_truth = (misc.imread(os.path.join(flags.img_path, flags.testing_data, valid_file.format('_truth_1feet.png'))) / 255).astype(np.uint8)

            if flags.is_CORAL:
                train_image = misc.imread(os.path.join(flags.img_path, flags.training_data, '{}_01_RGB.png'.format(flags.training_data)))
                valid_image = image_adapt(valid_image, train_image, 1)

            valid_image = misc.imresize(valid_image, flags.resolution_ratio, interp='bilinear')
            valid_truth = misc.imresize(valid_truth, flags.resolution_ratio, interp='nearest')

            if flags.is_mean_transfer:
                IMG_MEAN = np.mean(valid_image, axis=(0, 1))  # Image mean of testing data

            valid_image = valid_image - IMG_MEAN  # substract mean from image

            image_shape = valid_truth.shape

            valid_patches = patchify(valid_image, flags.image_size, valid_stride)
            """divided patches into smaller batch for evaluation"""
            pred_pmap = valid_in_batch(valid_patches, sess, pred, image_batch, step=flags.batch_size)

            # pred_pmap = np.ones(valid_patches.shape[0:-1])

            print("Stiching patches")
            pred_pmap_weighted = pred_pmap * gfilter[None, :, :]
            pred_pmap_weighted_large = unpatchify(pred_pmap_weighted, image_shape, valid_stride)
            gauss_mask_large = unpatchify(np.ones(pred_pmap.shape) * gfilter[None, :, :], image_shape, valid_stride)
            pred_pmap_weighted_large_normalized = np.nan_to_num(pred_pmap_weighted_large / gauss_mask_large)
            pred_binary = (pred_pmap_weighted_large_normalized > flags.pred_threshold).astype(np.uint8)

            # mean IoU
            seg_metric.add_image_pair(pred_binary, valid_truth)
            message_temp = "{}, {:.4f}".format(valid_file[0:-2], mean_IU(pred_binary, valid_truth))
            print(message_temp)
            file.write(message_temp + '\n')

            print("Saving evaluation prediction")

            # misc.imsave(os.path.join(flags.save_dir, '{}_{}pred.png'.format(valid_file[0:-2], 'NT_' if not flags.is_mean_transfer else '')), pred_binary)
            misc.imsave(os.path.join(flags.save_dir, '{}_pred_threshold_{}{}{}.png'.format(valid_file[0:-2],flags.pred_threshold, '_TM' if flags.is_mean_transfer else '', '_CORAL' if flags.is_CORAL else '')), pred_binary * 255)
            misc.toimage(pred_pmap_weighted_large_normalized.astype(np.float32), high=1.0, low=0.0, cmin=0.0, cmax=1.0, mode='F').save(os.path.join(flags.save_dir, '{}_pred_pmap{}{}.tif'.format(valid_file[0:-2], '_TM' if flags.is_mean_transfer else '', '_CORAL' if flags.is_CORAL else '')))



        message_overall = "Overall, {:.4f}".format(seg_metric.mean_IU())
        print(message_overall)
        file.write(message_overall + '\n')
        file.close()
        print('The output file has been saved to {}'.format(flags.save_dir))

        sess.close()



if __name__ == '__main__':
    flags = read_flags()
    main(flags)
