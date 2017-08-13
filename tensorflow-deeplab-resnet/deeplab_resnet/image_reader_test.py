import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    # img = tf.image.resize_bilinear(tf.expand_dims(img, 0), new_shape)
    # img = tf.squeeze(img, squeeze_dims=[0])
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return img, label

def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label

def image_flipping(img, label):
    """
    randomly flips images left-right and up-down
    :param img:
    :param label:
    :return:flipped images
    """
    img = tf.image.random_flip_left_right(tf.image.random_flip_up_down(img))
    label = tf.image.random_flip_left_right(tf.image.random_flip_up_down(label))
    return img, label

def image_rotating(img, label):
    """
    randomly rotate images by 0/90/180/270 degrees
    :param img:
    :param label:
    :return:rotated images
    """
    random_times = tf.to_int32(tf.random_uniform([1], minval=0, maxval=4))[0]
    img = tf.image.rot90(img, random_times)
    label = tf.image.rot90(label, random_times)
    return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label]) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop  

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, random_rotate, ignore_label, img_mean): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)

    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        h, w = input_size

        # Randomly mirror the images and labels.
        # if random_mirror:
        img, label = image_flipping(img, label)

        # if random_rotate:
        img, label = image_rotating(img, label)

        # Randomly scale the images and labels.
        # if random_scale:
        img, label = image_scaling(img, label)

        # Randomly crops the images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    return img, label


def dequeue(self, num_elements):
    '''Pack images and labels into a batch.
    
    Args:
      num_elements: the batch size.
      
    Returns:
      Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
    image_batch, label_batch = tf.train.batch([image, label], num_elements)
    return image_batch, label_batch


# setup used GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

city_name = 'Norfolk'
data_dir = '/home/helios/Documents/data/building/{0}/{0}_01_patches_size128'.format(city_name)
data_list = '../dataset/{}_01.txt'.format(city_name)
input_size = (128,128)
img_mean = np.array((121.68045527, 132.14961763, 129.30317439), dtype=np.float32)
img_mean = np.array((127.07435926, 129.40160709, 128.28713284), dtype=np.float32) # mean of building data in BGR order

ignore_label=255

tf.set_random_seed(1234)
coord = tf.train.Coordinator()
image_list, label_list = read_labeled_image_list(data_dir, data_list)
images = tf.convert_to_tensor(image_list, dtype=tf.string)
labels = tf.convert_to_tensor(label_list, dtype=tf.string)
queue = tf.train.slice_input_producer([images, labels], shuffle=True)  # not shuffling if it is val
# image, label = read_images_from_disk(queue, input_size, random_scale=True, random_mirror=True,random_rotate=True, ignore_label=255, img_mean=img_mean)

"""read_images_from_disk"""
img_contents = tf.read_file(queue[0])
label_contents = tf.read_file(queue[1])

img_rgb = tf.image.decode_image(img_contents, channels=3)
img_rgb = tf.cast(img_rgb, dtype=tf.float32)
img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img_rgb)
img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
# Extract mean.
# img -= img_mean

label = tf.image.decode_image(label_contents, channels=1)

if input_size is not None:
    h, w = input_size

    # Randomly mirror the images and labels.
    # if random_mirror:
    # img, label = image_flipping(img, label)
    label = tf.cast(label, dtype=tf.float32)
    temp = tf.concat([img, label], axis=2)
    temp_flipped = tf.image.random_flip_left_right(tf.image.random_flip_up_down(temp))
    img = tf.slice(temp_flipped, [0,0,0], [-1,-1, 3])
    label = tf.slice(temp_flipped, [0,0,3], [-1,-1, 1])

#     # if random_rotate:
    img, label = image_rotating(img, label)
#
#     # Randomly scale the images and labels.
#     # if random_scale:
    img_s, label_s = image_scaling(img, label)
#
#     # Randomly crops the images and labels.
    img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)
#
image_batch, label_batch, imageFile_batch, labelFile_batch = tf.train.shuffle_batch([img, label, queue[0], queue[1]], 20, capacity=1060, min_after_dequeue=1000)
# imgFile, labelFile = tf.train.batch([queue[0], queue[1]], 20)
# Set up tf session and initialize variables.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

threads = tf.train.start_queue_runners(coord=coord, sess=sess)

LABEL, IMAGE, IMAGEFILE, LABELFILE = sess.run([label_batch, image_batch, imageFile_batch, labelFile_batch])

i=4
plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.squeeze(IMAGE[i, :, :, :]))
plt.subplot(1,2,2)
plt.imshow(np.squeeze(LABEL[i, :, :, :]))
plt.show()
pass
