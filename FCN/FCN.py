from __future__ import print_function

import datetime
import os
import sys

import scipy.misc as misc
import tensorflow as tf
from six.moves import xrange

import TensorflowUtils as utils

sys.path.append(os.path.realpath('../metrics'))
sys.path.append(os.path.realpath('../dataReader'))
sys.path.append(os.path.realpath('../utils'))

from mat_batch_reader import BatchReader
from seg_metric import SegMetric
from utils.patchify import patchify, unpatchify, gauss2D
from eval_segm import *

dataDir = os.path.expanduser("~/Documents/data/igarssTrainingAndTestingData")
matDir = os.path.join(dataDir, "training_patches_for_segmentation")
imgDir = os.path.join(dataDir, "imageFiles")
# import segmentation evaluation module
# sys.path.append(os.path.realpath('../py_img_seg_eval'))


# default FLAGS
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "128", "batch size for training")
tf.flags.DEFINE_integer("epochs", "1", "number of training epochs")
tf.flags.DEFINE_integer("visualize_size", "20", "batch size for visualizing results")
tf.flags.DEFINE_string("logs_dir", "logs", "path to logs directory")
# tf.flags.DEFINE_string("data_dir", "/home/helios/Documents/data/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("mat_dir", matDir, "path to mat files")
tf.flags.DEFINE_float("learning_rate", "1e-7", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "/home/helios/Documents/data/Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "validate", "Mode train/ test/ visualize")
tf.flags.DEFINE_string("GPU", "0", "path to logs directory")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSES = 2
IMAGE_SIZE = 41
TRAINING_FILES = xrange(1, 13)
VALIDATION_FILES = xrange(13, 14)


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob, mean_image):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :param image_mean: mean of training images, subtracted from all images
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    # mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_image)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSES], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSES], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSES], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSES])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSES], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
        pmap = tf.nn.softmax(conv_t3, name="probability_map")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3, pmap


def train(loss_val, var_list, global_step):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step=global_step)


def test_in_batch(patches, session, out_var, input_var, keep_pro_var, step=100):
    num_patches = patches.shape[0]
    print("Testing total {} patch".format(num_patches))
    output = []
    for itr in range(0, int(np.ceil(num_patches / step))):
        temp_output = session.run(out_var,
                                  feed_dict={input_var: patches[itr * step:min((itr + 1) * step, num_patches), :, :, :],
                                             keep_pro_var: 1.0})
        if itr == 0:
            output = temp_output[:, :, :, 1]
        else:
            output = np.concatenate((output, temp_output[:, :, :, 1]), axis=0)
        print("Tested {} to {} patches".format(itr * step, min((itr + 1) * step, num_patches)))

    return output


def main(argv=None):
    # GPU setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU

    # log folder
    log_dir = os.path.join(FLAGS.logs_dir, "logs_batch{}/".format(FLAGS.batch_size))
    log_image_dir = log_dir + "images"

    print("Setting up dataset reader")
    # image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    # if (FLAGS.mode == 'train') | (FLAGS.mode == 'visualize'):
    if FLAGS.mode == 'train':
        train_dataset_reader = BatchReader(FLAGS.mat_dir, TRAINING_FILES)
        train_dataset_reader.shuffle_images()
    # if (FLAGS.mode == 'train')| (FLAGS.mode == 'visualize')| (FLAGS.mode == 'test'):
    validation_dataset_reader = BatchReader(FLAGS.mat_dir, VALIDATION_FILES)

    print("Setting up Graph")
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    pred_annotation, logits, pmap = inference(image, keep_probability, validation_dataset_reader.mean_image)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation,
                                                                                            squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var, global_step)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    # print("Setting up image reader...")
    # train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    # print(len(train_records))
    # print(len(valid_records))


    # GPU configuration to avoid that TF takes all GPU memeory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=3)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from {}".format(log_dir))

    if FLAGS.mode == "train":
        print("Start training with batch size {}, learning rate{}".format(FLAGS.batch_size, FLAGS.learning_rate))
        itr = int(0)
        while train_dataset_reader.epochs_completed < FLAGS.epochs:
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            _, itr = sess.run([train_op, global_step], feed_dict=feed_dict)

            if itr % 20 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("EPOCH %d Step: %d, Train_loss:%g" % (train_dataset_reader.epochs_completed, itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 200 == 0:
                valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, log_dir + "model.ckpt", itr)
                print("Checkpoint saved")
        saver.save(sess, log_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        print("visualize {} images".format(FLAGS.visualize_size))
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.visualize_size)
        # train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.visualize_size)

        pred_valid, probability_map = sess.run([pred_annotation, pmap], feed_dict={image: valid_images,
                                                                                   keep_probability: 1.0})
        # pred_train = sess.run(pred_annotation, feed_dict={image: train_images, annotation: train_annotations, keep_probability: 1.0})

        valid_annotations = np.squeeze(valid_annotations, axis=3)
        # train_annotations = np.squeeze(train_annotations, axis=3)
        pred_valid = np.squeeze(pred_valid, axis=3)
        # pred_train = np.squeeze(pred_train, axis=3)
        probability_map = probability_map[:, :, :, 1]

        """save images"""
        log_image_dir = log_dir + "images"
        if not os.path.isdir(log_image_dir):
            os.makedirs(log_image_dir)

        for itr in range(FLAGS.visualize_size):
            utils.save_image(valid_images[itr].astype(np.uint8), log_image_dir, name="inp_test" + str(itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), log_image_dir, name="gt_test" + str(itr))
            utils.save_image(pred_valid[itr].astype(np.uint8), log_image_dir, name="pred_test" + str(itr))
            utils.save_image(probability_map[itr].astype(np.double), log_image_dir, name="pmap_test" + str(itr))
            print("Saved image: %d" % itr)

            # for itr in range(FLAGS.visualize_size):
            #     utils.save_image(train_images[itr].astype(np.uint8), imageDir, name="inp_train" + str(itr))
            #     utils.save_image(train_annotations[itr].astype(np.uint8), imageDir, name="gt_train" + str(itr))
            #     utils.save_image(pred_train[itr].astype(np.uint8), imageDir, name="pred_train" + str(itr))
            #     print("Saved image: %d" % itr)

    elif FLAGS.mode == "validate":
        valid_stride = 20
        valid_batch_size = 1000
        valid_list = ['11ska595800{}', '11ska460755{}', '11ska580860{}', '11ska565845{}']

        gfilter = gauss2D(shape=[IMAGE_SIZE, IMAGE_SIZE], sigma=(IMAGE_SIZE - 1) / 4)

        for valid_file in valid_list:
            print("Validate image {}".format(valid_file[0:-2]))
            valid_image = misc.imread(os.path.join(imgDir, valid_file.format('.png')))
            valid_annotation = misc.imread(os.path.join(imgDir, valid_file.format('_truth.png')))
            image_shape = valid_annotation.shape

            valid_patches = patchify(valid_image, IMAGE_SIZE, valid_stride)

            """divided patches into smaller batch for validation"""
            pred_pmap = test_in_batch(valid_patches, sess, pmap, image, keep_probability, step=valid_batch_size)

            pred_pmap_weighted = pred_pmap * gfilter[None, :, :]
            pred_weighted_rec = unpatchify(pred_pmap_weighted, image_shape, valid_stride)
            gauss_mask_rec = unpatchify(np.ones(pred_pmap.shape) * gfilter[None, :, :], image_shape, valid_stride)
            pred_weighted_normalized = np.nan_to_num(pred_weighted_rec / gauss_mask_rec)

            print("Save validation prediction")
            utils.save_image(pred_weighted_normalized.astype(np.float32), log_image_dir,
                             name="{}_valid_pred".format(valid_file[0:-2]))
            misc.toimage(pred_weighted_normalized.astype(np.float32), high=1.0, low=0.0, cmin=0.0, cmax=1.0,
                         mode='F').save(os.path.join(log_image_dir, '{}_valid_pmap.tif'.format(valid_file[0:-2])))

            print("mean_IU: {}".format(mean_IU((pred_weighted_normalized > 0.5).astype(int), valid_annotation)))


    elif FLAGS.mode == "test":
        """
        test on validation images one by one
        """
        # for itr in xrange(len(valid_records)):
        #     valid_images, valid_annotations = validation_dataset_reader.next_batch(1)
        #     pred_valid = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
        #                                             keep_probability: 1.0})
        #     valid_annotations = np.squeeze(valid_annotations)
        #     pred_valid = np.squeeze(pred_valid)
        #     if itr == 0:
        #         valid_annotations_concatenate = valid_annotations
        #         pred_concatenate = pred_valid
        #     else:
        #         valid_annotations_concatenate = np.concatenate((valid_annotations_concatenate,valid_annotations),axis = 0)
        #         pred_concatenate = np.concatenate((pred_concatenate, pred_valid),axis = 0)
        #     print('test %d th image of %d validation image' %(itr+1,len(valid_records)))
        #
        # print(pixel_accuracy(valid_annotations_concatenate, pred_concatenate))
        # print(mean_accuracy(valid_annotations_concatenate, pred_concatenate))
        # print(mean_IU(valid_annotations_concatenate, pred_concatenate))
        # print(frequency_weighted_IU(valid_annotations_concatenate, pred_concatenate))

        """save prediction results on validation images"""
        print("testing validation images")
        seg_metric = SegMetric(NUM_OF_CLASSES)
        for itr in range(validation_dataset_reader.images.shape[0]):
            print("testing {}th image".format(itr + 1))
            valid_images, valid_annotations = validation_dataset_reader.next_batch(1)
            pred_valid = sess.run(pred_annotation,
                                  feed_dict={image: valid_images, annotation: valid_annotations, keep_probability: 1.0})
            valid_annotations = np.squeeze(valid_annotations)
            pred_valid = np.squeeze(pred_valid)
            seg_metric.add_image_pair(pred_valid, valid_annotations)
            if (itr + 1) % 1000 == 0:
                print("itr{}:".format(itr + 1))
                seg_metric.pixel_accuracy()
                seg_metric.mean_accuracy()
                seg_metric.mean_IU()
                seg_metric.frequency_weighted_IU()

        print("Final Accuracy:")
        seg_metric.pixel_accuracy()
        seg_metric.mean_accuracy()
        seg_metric.mean_IU()
        seg_metric.frequency_weighted_IU()


if __name__ == "__main__":
    tf.app.run()
