import tensorflow as tf
import scipy.misc as misc
import os
import numpy as np
from skimage.util.shape import view_as_windows


def gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def patchify(image, patch_size, step):
    """
    A function that devide a big image into small patches with mesh grid (patches could be overlapped)

    Parameters: 
        image: numpy array [m, n, 3] (RGB) or [m, n] (gray scale)
        patch_size: scalar
        step: scalar, the distance between centers of two adjacent patch is step

    Return:
        A tensor with shape of [number_of_patches_row, number_of_patches_col, patch_size, patch_size, 3] 
    """
    if image.shape.__len__() == 3:
        window_shape = [patch_size, patch_size, image.shape[-1]]
        step = [step, step, image.shape[-1]]
        return np.reshape(view_as_windows(image, window_shape, step=step),
                          [-1, patch_size, patch_size, image.shape[-1]])
    if image.shape.__len__() == 2:
        window_shape = [patch_size, patch_size]
        step = [step, step]
        return np.reshape(view_as_windows(image, window_shape, step=step),
                          [-1, patch_size, patch_size])


def unpatchify(patches, image_shape, step):
    # size of one patch, assuming any patch is a square
    patch_size = patches.shape[1]

    # number of patches in each row and col
    num_patch = [int((image_shape[i] - patch_size) / step + 1) for i in range(0, 2)]
    image_merged = np.zeros(shape=image_shape, dtype=patches.dtype)

    for i in range(0, num_patch[0]):
        for j in range(0, num_patch[1]):

            if image_shape.__len__() == 3:
                image_merged[i * step:(i * step + patch_size), j * step:(j * step + patch_size), :] += patches[
                                                                                                       i * num_patch[
                                                                                                           1] + j, :, :,
                                                                                                       :]
                # To Do: figure out the zero problem
                # if np.any(patches[i*num_patch[0]+j, :, :, :])

            elif image_shape.__len__() == 2:
                image_merged[i * step:(i * step + patch_size), j * step:(j * step + patch_size)] += patches[
                                                                                                    i * num_patch[
                                                                                                        1] + j, :, :]

    return image_merged


def patchify_balanced(image, truth, num_patch, patch_shape, ratio_foreground=0.5, foreground_threshold=0.1):
    """
    patchify a large images into small patches. The patches are randomly cropped from the large image
    and the number of foreground patches and background patches are balanced into ratio_foreground. If a patch is foreground
    is decided by if the ratio of foreground pixels in that patch is more than foreground_threshold
    :param image: (M, N, 3) array
    :param truth: (M, N) array with labels
    :param num_patch: int, number  
    :param patch_shape: A tuple with 2 element
    :param ratio_foreground: 
    :param foreground_threshold: 
    :return: 
    """

    # valid shape of large image that can be used as the left_top_corner of the cropping window
    valid_shape = np.subtract(image.shape[0:2], patch_shape)

    # seeds that generated from all possible window positions
    seeds = np.random.choice(valid_shape[0] * valid_shape[1], num_patch * 10)

    image_patches = np.zeros(shape=(num_patch, patch_shape[0], patch_shape[1], 3), dtype=image.dtype)
    truth_patches = np.zeros(shape=(num_patch, patch_shape[0], patch_shape[1]), dtype=truth.dtype)

    num_patch_foreground = num_patch * ratio_foreground
    num_patch_background = num_patch * (1 - ratio_foreground)

    ii = 0
    jj = 0

    for seed in seeds:
        # row and coloumn indices of left top corner of cropping window
        win_rol = int(np.floor(seed / valid_shape[1]))
        win_col = int(seed % valid_shape[1])
        img_patch_temp = img_crop(image, (win_rol, win_col), patch_shape)
        truth_patch_temp = img_crop(truth, (win_rol, win_col), patch_shape)

        ratio = np.sum(truth_patch_temp > 0) / (patch_shape[0] * patch_shape[1])
        if ratio <= foreground_threshold and ii < num_patch_background:
            image_patches[ii + jj] = img_patch_temp
            truth_patches[ii + jj] = truth_patch_temp
            ii += 1

        if ratio > foreground_threshold and jj < num_patch_foreground:
            image_patches[ii + jj] = img_patch_temp
            truth_patches[ii + jj] = truth_patch_temp
            jj += 1

    return image_patches, truth_patches


def img_crop(img, left_top_corner, window_shape):
    """
    Crop a image with a window with shape window_shape, of which the left top corner is given
    :param img: a 2D or 3D  numpy array, the input image
    :param left_top_corner: a tuple, left top corner of cropping window (row, col)
    :param window_shape: a tuple, shape of window (row, col)
    :return: cropped image
    """

    right_bottow_row = left_top_corner[0] + window_shape[0]
    right_bottow_col = left_top_corner[1] + window_shape[1]

    if left_top_corner[0] < 0 or left_top_corner[1] < 0 or right_bottow_row >= img.shape[
        0] or right_bottow_col >= img.shape[1]:
        raise RuntimeError("Crop outside of image")

    return img[left_top_corner[0]:right_bottow_row, left_top_corner[1]:right_bottow_col]


#
# patch_size = 41
# step = 20
# imageDir = os.path.expanduser('~/Documents/data/igarssTrainingAndTestingData/imageFiles')
# imageFileName = '11ska595800{}'
# one_image = misc.imread(os.path.join(imageDir,imageFileName.format('_truth.png')))
# image_size = one_image.shape
# # image4D = one_image.reshape((1,)+one_image.shape)
# patches = patchify(one_image, patch_size,step)
#
# gfilter = gauss2D(shape=[patch_size,patch_size], sigma=(patch_size-1)/4)
#
# # patches_weighted = patches*gfilter[None, :, :, None]
# patches_weighted = patches*gfilter[None, :, :]
#
# image_weighted_rec = unpatchify(patches_weighted, image_size, step)
# # gfilter_rec = unpatchify(np.ones(patches.shape)*gfilter[None, :, :, None], image_size, step)
# gfilter_rec = unpatchify(np.ones(patches.shape)*gfilter[None, :, :], image_size, step)
#
# image_weighted_normalized = image_weighted_rec/gfilter_rec
#
# plt.figure()
# plt.subplot(221)
# plt.imshow(one_image)
# plt.subplot(222)
# plt.imshow(image_weighted_rec)
# plt.subplot(223)
# plt.imshow(gfilter_rec,cmap = plt.cm.gray)
# plt.subplot(224)
# plt.imshow(np.uint8(image_weighted_normalized))
# plt.show()
# pass


"""solution using tensorflow not working"""
# c = 3
# h = 5000
# p = int(20)
#
# # Image to Patches Conversion
# pad = [[0,0],[0,0]]
# patches = tf.space_to_batch_nd(image4D,[p,p],pad)
# patches = tf.split(patches,p*p,0)
# patches = tf.stack(patches,3)
# patches = tf.reshape(patches,[(h/p)**2,p,p,c])
#
# # Do processing on patches
# # Using patches here to reconstruct
# patches_proc = tf.reshape(patches,[1,h/p,h/p,p*p,c])
# patches_proc = tf.split(patches_proc,p*p,3)
# patches_proc = tf.stack(patches_proc,axis=0)
# patches_proc = tf.reshape(patches_proc,[p*p,h/p,h/p,c])
#
# reconstructed = tf.batch_to_space_nd(patches_proc,[p, p],pad)
#
# sess = tf.Session()
# I,P,R_n = sess.run([image,patches,reconstructed])
# print(I.shape)
# print(P.shape)
# print(R_n.shape)
# err = np.sum((R_n-I)**2)
# print(err)

"""solution using numpy directly"""
# x = one_image[:,:,0]
# indices = np.array([[20,20,20],[21,31,41]])
# patch_size = 40
#
# m,n = x.shape
# K = int(np.floor(patch_size/2.0))
# R = np.arange(-K,K+1)
# patches = np.take(x,R[:,None]*n + R + (indices[0]*n+indices[1])[:,None,None])

"""solution using view window"""
