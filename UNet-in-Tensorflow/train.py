# -*- coding: utf-8 -*-
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
sys.path.append(os.path.realpath('../dataReader'))
from image_reader import ImageReader
sys.path.append(os.path.realpath('../metrics'))
from seg_metric import SegMetric
from utils import decode_labels, inv_preprocess, prepare_label
import numpy as np


TRAINING_DATA = "all"
# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) # original mean
IMG_MEAN = np.array((121.68045527, 132.14961763, 129.30317439), dtype=np.float32) # mean of solar panel data in BGR order
# IMG_MEAN = np.array((127.07435926, 129.40160709, 128.28713284), dtype=np.float32) # mean of building data in BGR order
BATCH_SIZE = 10
# DATA_DIRECTORY = '/home/helios/Documents/data/PASCAL_VOC2012'
DATA_DIRECTORY = "/home/helios/Documents/data/building/Norfolk/Norfolk_01_patches_size128"
DATA_LIST_PATH = './dataset/train_sc.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '128,128'
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_CLASSES = 2
EPOCHS = 1
DATA_SIZE = 100
NUM_STEPS = int(20*22363/BATCH_SIZE)
DECAY_STEP = 0
DECAY_RATE = 0.1
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/home/helios/Documents/data/Model_zoo/deeplab_resnet_init.ckpt'
SNAPSHOT_DIR = './snapshots'
SAVE_NUM_IMAGES = 10
SAVE_PRED_EVERY = 1000
GPU = '0'

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


def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",
                        default=30,
                        type=int,
                        help="Number of epochs (default: 1)")

    parser.add_argument("--batch-size",
                        default=20,
                        type=int,
                        help="Batch size (default: 20)")

    parser.add_argument("--ckdir",
                        default="models",
                        help="Checkpoint directory (default: models)")

    parser.add_argument("--training-data", type=str, default=TRAINING_DATA,
                        help="which data used for training: SP - solar panel data; $cityname$ - building data at $cityname$")

    parser.add_argument("--training-data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")

    parser.add_argument("--validation-data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")

    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")

    parser.add_argument("--is-loss-entropy", action="store_true",
                        help="Whether to use cross entropy as loss function (otherwise use IoU).")

    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")


    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--decay-step", type=float, default=DECAY_STEP,
                        help="Learning rate decay step in number of epochs.")
    parser.add_argument("--decay-rate", type=float, default=DECAY_RATE,
                        help="Learning rate decay rate")


    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-rotate", action="store_true",
                        help="Whether to randomly rotate the inputs during the training.")

    parser.add_argument("--GPU", type=str, default=GPU,
                        help="GPU used for computation.")

    flags = parser.parse_args()
    return flags

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

def image_summary(image, truth, prediction, image_mean, image_std=None, num_classes=2, max_output=10):
    """

    :param image: 4-D array(N, H, W, 3)
    :param truth: 4-D array(N, H, W, 1)
    :param prediction: 4-D array(N, H, W, 1)
    :param image_mean: [B,G,R]
    :param image_std: [B,G,R]
    :param num_classes: scalar
    :param max_output: scalar, must be less than N
    :return: 4-D array(max_output, H, 3*W, 3)
    """
    images = inv_preprocess(image, max_output, image_mean, image_std)
    labels = decode_labels(truth, max_output, num_classes)
    predictions = decode_labels(prediction, max_output, num_classes)

    return np.concatenate([images, labels, predictions], axis=2)

def main(flags):
    IMG_MEAN = np.zeros(3)
    image_std = [1.0, 1.0, 1.0]
    # parameters of building data set
    citylist = ['Norfolk', 'Arlington', 'Atlanta', 'Austin', 'Seekonk', 'NewHaven']
    image_mean_list = {'Norfolk': [127.07435926, 129.40160709, 128.28713284],
                       'Arlington': [88.30304996, 94.97338776, 93.21268212],
                       'Atlanta': [101.997014375, 108.42171833, 110.044871],
                       'Austin': [97.0896012682, 102.94697026, 100.7540157],
                       'Seekonk': [86.67800904, 93.31221168, 92.1328146],
                       'NewHaven': [106.7092798, 111.4314,
                                    110.74903832]}  # BGR mean for the training data for each city

    # set training data
    if flags.training_data == 'SP':
        IMG_MEAN = np.array((121.68045527, 132.14961763, 129.30317439),
                            dtype=np.float32)  # mean of solar panel data in BGR order


    elif flags.training_data in citylist:
        print("Training on {} data".format(flags.training_data))
        IMG_MEAN = image_mean_list[flags.training_data]
        # if flags.unit_std:
        #     image_std = image_std_list[flags.training_data]
    elif 'all_but' in flags.training_data:
        print("Training on all(excludes Seekonk) but {} data".format(flags.training_data))
        except_city_name = flags.training_data.split('_')[2]
        for cityname in citylist:
            if cityname != except_city_name and cityname != 'Seekonk':
                IMG_MEAN = IMG_MEAN + np.array(image_mean_list[cityname])
        IMG_MEAN = IMG_MEAN/4

    elif flags.training_data == 'all':
        print("Training on data of all cities (excludes Seekonk)")
        for cityname in citylist:
            if cityname != 'Seekonk':
                IMG_MEAN = IMG_MEAN + np.array(image_mean_list[cityname])
        IMG_MEAN = IMG_MEAN / 5
    else:
        print("Wrong data option: {}".format(flags.data_option))

    # setup used GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.GPU

    # presetting
    input_size = (128, 128)
    tf.set_random_seed(1234)
    coord = tf.train.Coordinator()
    # img_mean = [127.07435926, 129.40160709, 128.28713284]
    with tf.name_scope("training_inputs"):
        training_reader = ImageReader(
            flags.training_data_list,
            input_size,
            random_scale=True,
            random_mirror=True,
            random_rotate=True,
            ignore_label=255,
            img_mean=IMG_MEAN,
            coord=coord
            )
    with tf.name_scope("validation_inputs"):
        validation_reader = ImageReader(
            flags.validation_data_list,
            input_size,
            random_scale=False,
            random_mirror=False,
            random_rotate=False,
            ignore_label=255,
            img_mean=IMG_MEAN,
            coord=coord,
            )
    X_batch_op, y_batch_op = training_reader.shuffle_dequeue(flags.batch_size)
    X_test_op, y_test_op = validation_reader.shuffle_dequeue(flags.batch_size*2)



    train = pd.read_csv(flags.training_data_list, header=0)
    n_train = train.shape[0]+1

    test = pd.read_csv(flags.validation_data_list, header=0)
    n_test = test.shape[0]+1

    current_time = time.strftime("%m_%d/%H_%M")

    # tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name="X")
    y = tf.placeholder(tf.float32, shape=[None, 128, 128, 1], name="y")
    mode = tf.placeholder(tf.bool, name="mode")

    pred_raw = make_unet(X, mode)
    pred = tf.nn.sigmoid(pred_raw)
    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)

    tf.summary.histogram("Predicted Mask", pred)
    # tf.summary.image("Predicted Mask", pred)

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


    learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step,
                                               tf.cast(n_train / flags.batch_size * flags.decay_step, tf.int32), flags.decay_rate, staircase=True)

    IOU_op = IOU_(pred, y)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred))
    tf.summary.scalar("loss/IOU_training", IOU_op)
    tf.summary.scalar("loss/cross_entropy_training", cross_entropy)

    learning_rate_summary = tf.summary.scalar("learning_rate", learning_rate)  # summary recording learning rate

    #loss = cross_entropy
    if  flags.is_loss_entropy:
        loss = cross_entropy
    else:
        loss = -IOU_op

    with tf.control_dependencies(update_ops):
        train_op = make_train_op(loss, global_step, learning_rate)
        # train_op = make_train_op(cross_entropy, global_step, learning_rate)


    summary_op = tf.summary.merge_all()

    valid_IoU = tf.placeholder(tf.float32, [])
    valid_IoU_summary_op = tf.summary.scalar("loss/IoU_validation",valid_IoU)
    valid_cross_entropy = tf.placeholder(tf.float32, [])
    valid_cross_entropy_summary_op = tf.summary.scalar("loss/cross_entropy_validation", valid_cross_entropy)

    # original images for summary
    train_images = tf.placeholder(tf.uint8, shape=[None, 128, 128 * 3, 3], name="training_images")
    train_image_summary_op = tf.summary.image("Training_images_summary", train_images, max_outputs=10)
    valid_images = tf.placeholder(tf.uint8, shape=[None, 128, 128 * 3, 3], name="validation_images")
    valid_image_summary_op = tf.summary.image("Validation_images_summary", valid_images, max_outputs=10)


    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        if os.path.exists(flags.ckdir) and tf.train.get_checkpoint_state(flags.ckdir):
            latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
            saver.restore(sess, latest_check_point)

        # elif not os.path.exists(flags.ckdir):
        #     # try:
        #     #     os.rmdir(flags.ckdir)
        #     # except FileNotFoundError:
        #     #     pass
        #     os.mkdir(flags.ckdir)

        try:
            train_summary_writer = tf.summary.FileWriter(flags.ckdir, sess.graph)

            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            for epoch in range(flags.epochs):

                for step in range(0, n_train, flags.batch_size):

                    start_time = time.time()
                    X_batch, y_batch = sess.run([X_batch_op, y_batch_op])

                    _,  global_step_value = sess.run([train_op, global_step],feed_dict={X: X_batch,y: y_batch,mode: True})
                    if global_step_value % 100 == 0:
                        duration = time.time() - start_time
                        pred_train, step_iou, step_cross_entropy,  step_summary, = sess.run([
                            pred, IOU_op, cross_entropy, summary_op],
                            feed_dict={X: X_batch,
                                       y: y_batch,
                                       mode: False})
                        train_summary_writer.add_summary(step_summary, global_step_value)

                        print('Epoch {:d} step {:d} \t cross entropy = {:.3f}， IOU = {:.3f} ({:.3f} sec/step)'.format(epoch, global_step_value, step_cross_entropy, step_iou, duration))

                # validation every epoch
                    if global_step_value % 1000 == 0:
                        segmetric = SegMetric(1)
                        # for step in range(0, n_test, flags.batch_size):
                        X_test, y_test = sess.run([X_test_op, y_test_op])
                        pred_valid, valid_cross_entropy_value= sess.run(
                            [pred, cross_entropy],
                            feed_dict={X: X_test,
                                       y: y_test,
                                       mode: False})
                        iou_temp = myIOU(y_pred=pred_valid>0.5, y_true=y_test, segmetric=segmetric)
                        print("Test IoU: {}  Cross_Entropy: {}".format(segmetric.mean_IU(), valid_cross_entropy_value))

                        valid_IoU_summary = sess.run(valid_IoU_summary_op, feed_dict={valid_IoU: iou_temp})
                        train_summary_writer.add_summary(valid_IoU_summary, global_step_value)
                        valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op, feed_dict={valid_cross_entropy: valid_cross_entropy_value})
                        train_summary_writer.add_summary(valid_cross_entropy_summary, global_step_value)

                        train_image_summary = sess.run(train_image_summary_op,
                                                       feed_dict={train_images: image_summary(X_batch, y_batch,pred_train > 0.5, IMG_MEAN, num_classes=flags.num_classes)})
                        train_summary_writer.add_summary(train_image_summary, global_step_value)
                        valid_image_summary = sess.run(valid_image_summary_op,
                                                       feed_dict={valid_images: image_summary(X_test, y_test, pred_valid > 0.5, IMG_MEAN, num_classes=flags.num_classes)})
                        train_summary_writer.add_summary(valid_image_summary, global_step_value)
                    # total_iou += step_iou * X_test.shape[0]
                        #
                        # test_summary_writer.add_summary(step_summary, (epoch + 1) * (step + 1))


                saver.save(sess, "{}/model.ckpt".format(flags.ckdir), global_step=global_step)

        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(flags.ckdir),global_step=global_step)


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
