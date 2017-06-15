import pickle
import numpy as np
from eval_segm import *
from seg_metric import SegMetric


def sum_nan(a):
    return sum(np.nan_to_num(a))


with open('prediction.pickle','rb') as f:
    [pred, gt] = pickle.load(f)
    pred = np.squeeze(pred)
print(pred.shape)
print(gt.shape)

num_img = 10
for itr in range(num_img):
    valid_annotations = np.squeeze(gt[itr, :, :])
    pred_valid = np.squeeze(pred[itr, :, :])
    if itr == 0:
        valid_annotations_concatenate = valid_annotations
        pred_concatenate = pred_valid
    else:
        valid_annotations_concatenate = np.concatenate((valid_annotations_concatenate,valid_annotations),axis = 0)
        pred_concatenate = np.concatenate((pred_concatenate, pred_valid),axis = 0)
    # print('test %d th image of %d validation image' %(itr+1,pred.shape[0]))

print(valid_annotations_concatenate.shape, pred_concatenate.shape)
print(pixel_accuracy(pred_concatenate, valid_annotations_concatenate))
print(mean_accuracy(pred_concatenate, valid_annotations_concatenate))
print(mean_IU(pred_concatenate, valid_annotations_concatenate))
print(frequency_weighted_IU(pred_concatenate, valid_annotations_concatenate))

"""
Test using implemented class
"""

seg_metric = SegMetric(151)
for i in range(num_img):
    eval_segm = pred[i, :, :]
    gt_segm = gt[i, :, :]
    seg_metric.add_image_pair(eval_segm, gt_segm)

seg_metric.pixel_accuracy()
seg_metric.mean_accuracy()
seg_metric.mean_IU()
seg_metric.frequency_weighted_IU()

#
# """
# Test using new method
# """
#
#
# """initialize variables"""
# max_num_classes = 151
#
# current_num_classes = 0
# current_classes = np.array([], dtype=int)
#
# current_num_classes_gt = 0
# current_classes_gt = np.array([], dtype=int)
#
# n_ij = np.zeros(max_num_classes)  # number of pixels predicted as class j
# n_ii = np.zeros(max_num_classes)  # number of pixels predicted as class i correctly
# t_i = np.zeros(max_num_classes)   # number of pixels that are truly class i
#
#
# """Add image"""
#
# i=0
# eval_segm = pred[i, :, :]
# gt_segm = gt[i, :, :]
#
# check_size(eval_segm, gt_segm)  # sizes of two images have to be the same
#
# # get the existing classes from provided images
# temp_cl, temp_num_cl = union_classes(eval_segm, gt_segm)
# current_classes = np.union1d(current_classes, temp_cl)
# current_num_classes = current_classes.size
#
# temp_cl_gt, temp_num_cl_gt = extract_classes(gt_segm)
# current_classes_gt = np.union1d(current_classes_gt, temp_cl_gt)
# current_num_classes_gt = current_classes_gt.size
#
# #computer the binary mask for each classes
# eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, current_classes, current_num_classes)
#
# for i, c in enumerate(current_classes):
#     curr_eval_mask = eval_mask[i, :, :]
#     curr_gt_mask = gt_mask[i, :, :]
#
#     n_ii[c] = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
#     n_ij[c] = np.sum(curr_eval_mask)
#     t_i[c] = np.sum(curr_gt_mask)
#     # print(c, n_ii[c])
#     # print(c, n_ij[c])
#     # print(c, t_i[c])
#
# """Pixel Accuracy"""
# with np.errstate(divide='ignore', invalid='ignore'):
#     PA = np.sum(n_ii)/np.sum(t_i)
#     print(PA)
#
# """Mean Pixel Accuracy"""
# with np.errstate(divide='ignore',  invalid='ignore'):
#     MPA = sum_nan(n_ii / t_i) / current_num_classes_gt
#     print(MPA)
#
# """Mean Intersection over Union (Mean IU)"""
# with np.errstate(divide='ignore',  invalid='ignore'):
#     MIU = sum_nan(n_ii / (t_i + n_ij - n_ii)) / current_num_classes_gt
#     print(MIU)
#
# """Frequency Weighted IU"""
# with np.errstate(divide='ignore',  invalid='ignore'):
#     WIU = sum_nan(n_ii * t_i / (t_i + n_ij - n_ii))/sum_nan(t_i)
#     print(WIU)

