'''
metrics class for image segmentation evaluation
'''
import numpy as np
import pickle
import numpy as np


def sum_nan(a):
    return sum(np.nan_to_num(a))


class SegMetric:
    max_num_classes = 1

    current_num_classes = 0
    current_classes = np.array([], dtype=int)

    current_num_classes_gt = 0
    current_classes_gt = np.array([], dtype=int)

    def __init__(self, max_num_classes):
        self.max_num_classes = max_num_classes
        self.n_ij = np.zeros(max_num_classes)  # number of pixels predicted as class j
        self.n_ii = np.zeros(max_num_classes)  # number of pixels predicted as class i correctly
        self.t_i = np.zeros(max_num_classes)  # number of pixels that are truly class i

    """Add image pair"""
    def add_image_pair(self, eval_segm, gt_segm):
        check_size(eval_segm, gt_segm)  # sizes of two images have to be the same

        # get the existing classes from provided images
        temp_cl, temp_num_cl = union_classes(eval_segm, gt_segm)
        self.current_classes = np.union1d(self.current_classes, temp_cl)
        self.current_num_classes = self.current_classes.size

        # get the existing classes from provided images
        temp_cl_gt, temp_num_cl_gt = extract_classes(gt_segm)
        self.current_classes_gt = np.union1d(self.current_classes_gt, temp_cl_gt)
        self.current_num_classes_gt = self.current_classes_gt.size

        # computer the binary mask for each classes
        eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, temp_cl, temp_num_cl)

        for i, c in enumerate(temp_cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            self.n_ii[c] += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            self.n_ij[c] += np.sum(curr_eval_mask)
            self.t_i[c] += np.sum(curr_gt_mask)

    """Pixel Accuracy"""
    def pixel_accuracy(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            pixel_accuracy = np.sum(self.n_ii) / np.sum(self.t_i)
            print("pixel_accuracy:",pixel_accuracy)
            return pixel_accuracy

    """Mean Pixel Accuracy"""
    def mean_accuracy(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_accuracy = sum_nan(self.n_ii / self.t_i) / self.current_num_classes_gt
            print("Mean accuracy:",mean_accuracy)
            return mean_accuracy

    """Mean Intersection over Union (Mean IU)"""
    def mean_IU(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_IU = sum_nan(self.n_ii / (self.t_i + self.n_ij - self.n_ii)) / self.current_num_classes_gt
            print("mean_IU:", mean_IU)
            return mean_IU

    """Frequency Weighted IU"""
    def frequency_weighted_IU(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            frequency_weighted_IU = sum_nan(self.n_ii * self.t_i / (self.t_i + self.n_ij - self.n_ii)) / sum_nan(self.t_i)
            print("frequency_weighted_IU:", frequency_weighted_IU)
            return frequency_weighted_IU


"""
Following functions are borrowed from eval_seg 
at https://github.com/martinkersner/py_img_seg_eval
"""


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
