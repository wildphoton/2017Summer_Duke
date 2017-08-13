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

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) # original mean
IMG_MEAN = np.array((121.68045527, 132.14961763, 129.30317439), dtype=np.float32) # mean of solar panel data in BGR order

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
NUM_STEPS = int(20*22363/BATCH_SIZE)
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/home/helios/Documents/data/Model_zoo/deeplab_resnet_init.ckpt'
SNAPSHOT_DIR = './snapshots'
SAVE_NUM_IMAGES = 10
SAVE_PRED_EVERY = 1000
GPU = '0'
UP_SCALE = 8

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--up-scale", type=int, default=UP_SCALE,
                        help="the scale that upsample the input image")
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
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
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

def main():
    """Create the model and start the training."""
    args = get_arguments()

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
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)

    image_batch_up = tf.image.resize_bilinear(image_batch, [h*args.up_scale, w*args.up_scale])

    # Create network.
    net = DeepLabResNetModel({'data': image_batch_up}, is_training=args.is_training, num_classes=args.num_classes)
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
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    
    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())

    # learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    learning_rate = base_lr
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

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

    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)

    image_summary = tf.summary.image('images',
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                                     max_outputs=args.save_num_images)  # Concatenate row-wise.
    loss_summary = tf.summary.scalar("loss", reduced_loss)
    entropy_summary = tf.summary.scalar("entropy", tf.reduce_mean(loss))
    l2_loss_summary = tf.summary.scalar("L2_loss", tf.add_n(l2_losses))
    learning_rate_summary = tf.summary.scalar("learning_rate", learning_rate)  # summary recording learning rate

    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())

    print("Setting up summary op...")
    total_summary = tf.summary.merge_all()

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)

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
    else:
        print("No model found at{}".format(args.restore_from))

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Initial status
    loss_value, entropy_loss, summary, itr = sess.run([reduced_loss, tf.reduce_mean(loss), total_summary, global_step])
    print('step {:d} \t loss = {:.3f}, entropy_loss = {:.3f})'.format(itr, loss_value, entropy_loss))
    summary_writer.add_summary(summary, itr)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }

        _, itr= sess.run([train_op, global_step], feed_dict=feed_dict)

        # save summary file
        if itr % 100 == 0:
            duration = time.time() - start_time
            loss_value, entropy_loss, summary, itr = sess.run([reduced_loss, tf.reduce_mean(loss), total_summary, global_step])
            summary_writer.add_summary(summary, itr)
            print('step {:d} \t loss = {:.3f}, entropy_loss = {:.3f}, ({:.3f} sec/step)'.format(itr, loss_value, entropy_loss, duration))

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

if __name__ == '__main__':
    main()
