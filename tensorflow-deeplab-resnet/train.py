"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, inv_preprocess, prepare_label
sys.path.append(os.path.realpath('../metrics'))
from eval_segm import *
from seg_metric import SegMetric

TRAIN_DATA = "Norfolk"
# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) # original mean
IMG_MEAN = np.array((121.68045527, 132.14961763, 129.30317439), dtype=np.float32) # mean of solar panel data in BGR order
# IMG_MEAN = np.array((127.07435926, 129.40160709, 128.28713284), dtype=np.float32) # mean of building data in BGR order
BATCH_SIZE = 10
# DATA_DIRECTORY = '/home/helios/Documents/data/PASCAL_VOC2012'
DATA_DIRECTORY = '/home/helios/Documents/data/igarssTrainingAndTestingData/training_patches_images_size41'
DATA_LIST_PATH = './dataset/train_sc.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '41,41'
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

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--training-data", type=str, default=TRAIN_DATA,
                        help="which data used for training: SP - solar panel data; $cityname$ - building data at $cityname$")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    # parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
    #                     help="Number of training steps.")
    parser.add_argument("--num-epochs", type=int, default=EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--data-size", type=int, default=DATA_SIZE,
                        help="Number of training patches.")
    parser.add_argument("--decay-step", type=float, default=DECAY_STEP,
                        help="Learning rate decay step in number of epochs.")
    parser.add_argument("--decay-rate", type=float, default=DECAY_RATE,
                        help="Learning rate decay rate")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--unit-std", action="store_true",
                        help="Whether to make transfer training data to unit variance ")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-rotate", action="store_true",
                        help="Whether to randomly rotate the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--GPU", type=str, default=GPU,
                        help="GPU used for computation.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''Save weights.

   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def IOU_tf(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    # H, W, _ = y_pred.get_shape().as_list()[1:]
    #
    # pred_flat = tf.reshape(y_pred, [-1, H * W])
    # true_flat = tf.reshape(y_true, [-1, H * W])
    pred = tf.cast(y_pred, tf.int64)
    truth = tf.cast(y_true, tf.int64)

    # intersection = 2 *tf.cast( tf.reduce_sum(pred_flat * true_flat, axis=1), tf.float16) + 1e-7
    # denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7
    iou,_ = tf.metrics.mean_iou(labels=truth, predictions=pred, num_classes=2)
    # return tf.reduce_mean(intersection / denominator)
    return iou

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
    segmetric = SegMetric(1)
    for i in range(y_pred.shape[0]):
        segmetric.add_image_pair(y_pred[i,:,:,0], y_true[i,:,:,0])

    return segmetric.mean_IU()

def main():
    """Create the model and start the training."""
    args = get_arguments()

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
    image_std_list = {'Norfolk': [28.615469420031832, 32.662536832452886, 37.64149854207523],
                      'Arlington': [30.40903039206398, 37.973725024862595, 43.58402191634698],
                      'Atlanta': [36.93662467838125, 39.43470059838385, 41.74732676809388],
                      'Austin': [42.81337177109884, 43.71071321350751, 44.440517007230675],
                      'Seekonk': [25.506449467410715, 32.46885262572024, 37.76814267502958],
                      'NewHaven': [33.05784541012469, 36.62685162291547, 37.686084270914435]}

       # set evaluation data
    if args.training_data == 'SP':
        IMG_MEAN = np.array((121.68045527, 132.14961763, 129.30317439),
                            dtype=np.float32)  # mean of solar panel data in BGR order


    elif args.training_data in citylist:
        print("Training on {} data".format(args.training_data))
        IMG_MEAN = image_mean_list[args.training_data]
        if args.unit_std:
            image_std = image_std_list[args.training_data]
    else:
        print("Wrong data option: {}".format(args.data_option))

    # setup used GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    tf.set_random_seed(args.random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.random_rotate,
            args.ignore_label,
            IMG_MEAN,
            coord,
            image_std)
        image_batch, label_batch = reader.dequeue(args.batch_size)

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=args.is_training, num_classes=args.num_classes)
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))


    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + args.weight_decay * tf.add_n(l2_losses)

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    # step_ph = tf.placeholder(dtype=tf.float32, shape=())

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    global_epoch = tf.cast(global_step*args.batch_size / args.data_size, tf.int8)

    # learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

    if(args.decay_step<=0):
        learning_rate = base_lr
    else:
        learning_rate = tf.train.exponential_decay(base_lr, global_step, tf.cast(args.data_size/args.batch_size * args.decay_step, tf.int32), args.decay_rate, staircase=True)


    opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable), global_step=global_step)
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

    # IOU of training batch
    # IOU = IOU_tf(pred, label_batch)

    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN, image_std], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)

    image_summary = tf.summary.image('images',
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                                     max_outputs=args.save_num_images)  # Concatenate row-wise.
    loss_summary = tf.summary.scalar("loss", reduced_loss)
    entropy_summary = tf.summary.scalar("entropy", tf.reduce_mean(loss))
    l2_loss_summary = tf.summary.scalar("L2_loss", tf.add_n(l2_losses))
    learning_rate_summary = tf.summary.scalar("learning_rate", learning_rate)  # summary recording learning rate
    # IOU_summary = tf.summary.scalar("IOU", IOU)

    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())

    print("Setting up summary op...")
    total_summary = tf.summary.merge_all()

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2) # saver for save/load checkpoint

    # load weights from saved checkpoint or initial pre-trained model
    if os.path.isdir(args.snapshot_dir):
        # search checkpoint at given path
        ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # load checkpoint file
            load(saver, sess, ckpt.model_checkpoint_path)
    elif os.path.isfile(args.snapshot_dir):
        # load checkpoint file
        load(saver, sess, args.snapshot_dir)
    elif args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)  # loader for part of pre-trained model
        load(loader, sess, args.restore_from)
        print("Load weights from{}".format(args.restore_from) )
    else:
        print("No model found at{}".format(args.restore_from))

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Initial status
    loss_value, entropy_loss, summary, itr, epoch = sess.run([reduced_loss, tf.reduce_mean(loss), total_summary, global_step, global_epoch])
    print('step {:d} \t loss = {:.3f}, entropy_loss = {:.3f})'.format(itr, loss_value, entropy_loss))
    summary_writer.add_summary(summary, itr)

    # Iterate over training steps.
    while(epoch < args.num_epochs):
        start_time = time.time()
        # feed_dict = { step_ph : step }

        _, itr, epoch= sess.run([train_op, global_step, global_epoch])

        # save summary file
        if itr % 100 == 0:
            duration = time.time() - start_time
            pred_temp, truth_temp, loss_value, entropy_loss, summary, itr = sess.run([pred, label_batch, reduced_loss, tf.reduce_mean(loss), total_summary, global_step])
            summary_writer.add_summary(summary, itr)

            IOU_temp = IOU_(pred_temp, truth_temp)
            print('Epoch {:d} step {:d} \t loss = {:.3f}, entropy_loss = {:.3f}, IOU = {:.3f}, ({:.3f} sec/step)'.format(epoch, itr, loss_value, entropy_loss, IOU_temp, duration))
            # print('Epoch {:d} step {:d} \t loss = {:.3f}, entropy_loss = {:.3f}, ({:.3f} sec/step)'.format(
            # epoch, itr, loss_value, entropy_loss, duration))
        # save checkpoint
        if itr % args.save_pred_every == 0:
            # images, labels, preds = sess.run([image_batch, label_batch, pred])
            save(saver, sess, args.snapshot_dir, global_step)

    # final status
    loss_value, entropy_loss, summary, itr = sess.run([reduced_loss, tf.reduce_mean(loss), total_summary, global_step])
    print('step {:d} \t loss = {:.3f}, entropy_loss = {:.3f}'.format(itr, loss_value, entropy_loss))
    save(saver, sess, args.snapshot_dir, global_step)
    summary_writer.add_summary(summary, itr)

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()
