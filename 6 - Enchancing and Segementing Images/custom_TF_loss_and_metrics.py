"""
File name: custom_TF_loss_and_metrics.py
Author: YSLee
Date created: 30.07.2020
Date last modified:30.07.2020
Python Version: "3.7"
"""
#=========================== Import the Libraries ===========================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import tensorflow as tf
import cityscapesscripts.helpers.labels as cityscapes_labels

import numpy as np

#=========================== Constant Definitions ===========================
CITYSCAPES_FOLDER = os.getenv('CITYSCAPES_DATASET', default=os.path.expanduser('~/Dataset/cityscapes'))
# CITYSCAPES_FOLDER = os.path.expanduser('~/datasets/cityscapes')

CITYSCAPES_IGNORE_VALUE, value_to_ignore    = 255, 255
CITYSCAPES_LABELS           = [label for label in cityscapes_labels.labels
                               if -1 < label.trainId < (CITYSCAPES_IGNORE_VALUE or value_to_ignore)]
CITYSCAPES_COLOURS           = np.asarray( [label.color for label in CITYSCAPES_LABELS] )
CITYSCAPES_COLOURS_TF        = tf.constant(CITYSCAPES_COLOURS, dtype=tf.int32)
CITYSCAPES_IMG_RATIO        = 2
CITYSCAPES_INT_FILL         = 6
CITYSCAPES_FILE_TEMPLATE    = os.path.join(
    '{root}', '{type}', '{split}', '{city}',
    '{city}_{seq:{filler}>{len_fill}}_{frame:{filler}>{len_fill}}_{type}{type2}{ext}')

#=========================== loss and Metric Function Definitions ===========================

class SegmentationLoss(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, ignore_value=CITYSCAPES_IGNORE_VALUE, from_logits=False,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'):
        super().__init__(from_logits=from_logits, reduction=reduction, name=name)
        self.ignore_value = ignore_value

    def _prepare_data(self, y_true, y_pred):
        nb_classes = y_pred.shape[-1]

        y_true, y_pred = prepare_data_for_segmentation_loss(y_true=y_true,
                                                            y_pred=y_pred,
                                                            nb_classes=nb_classes,
                                                            ignore_value=self.ignore_value)

        return y_true, y_pred

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = self._prepare_data(y_true=y_true,
                                            y_pred=y_pred)
        loss = super().__call__(y_true, y_pred, sample_weight)

        return loss


class SegmentationAccuracy(tf.metrics.Accuracy):
    def __init__(self, ignore_value=CITYSCAPES_IGNORE_VALUE, name='acc', dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.ignore_value = ignore_value

    def __call__(self, y_true, y_pred, sample_weight=None):
        nb_classes = y_pred.shape[-1]

        y_true, y_pred = prepare_data_for_segmentation_loss(y_true=y_true,
                                                            y_pred=y_pred,
                                                            nb_classes=nb_classes,
                                                            ignore_value=self.ignore_value)

        # tf.metrics.Accuracy requires label maps in the original form and not one-hot encoded form:
        y_pred = tf.argmax(y_pred, axis=-1)

        return super().__call__(y_true, y_pred, sample_weight)


class SegmentationMeanIoU(tf.metrics.MeanIoU):
    def __init__(self, nb_classes, ignore_value=CITYSCAPES_IGNORE_VALUE, name='mIoU', dtype=None):
        super().__init__(num_classes=nb_classes, name=name, dtype=dtype)
        self.ignore_value = ignore_value
        self.num_classes = nb_classes

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = prepare_data_for_segmentation_loss(y_true=y_true,
                                                            y_pred=y_pred,
                                                            nb_classes=self.num_classes,
                                                            ignore_value=self.ignore_value)

        # tf.metrics.MeanIoU requires label maps in the original form and not one-hot encoded form:
        y_pred = tf.argmax(y_pred, axis=-1)

        return super().__call__(y_true, y_pred, sample_weight)

#=========================== Function Definitions ===========================

def get_mask_for_valid_labels(y_true, nb_classes, ignore_value=255):
    """ This builds the mask for the valid pixels.
    Parameters:
        - y_true, is the Ground-truth label map(s) where each value represents a class trainId.
        - nb_classes, is the total number of classes.
        - ignore_value, is the trainId class value to be ignored
    Returns:
        - returns the Binary mask """
    # valid classes:
    mask_for_class_elements = y_true < nb_classes

    # Ignored class:
    mask_for_not_ignored = y_true != ignore_value

    # Mask:
    mask = mask_for_class_elements & mask_for_not_ignored

    return mask


def prepare_data_for_segmentation_loss(y_true, y_pred, nb_classes=10, ignore_value=255):
    """ This functions prepares the predicted logits and ground-truth maps for the loss,
        removing the pixels from the ignored classes.
    Parameters:
        - y_true, is the Ground-truth label map(s), shape = (B x H x W).
        - y_pred, is the Predicted logit map(s), shape (B x H x W x N),
                    where N = number of classes.
        - nb_classes, is the number of classes.
        - ignore_value, is the trainId value of the ignored classes.
    Returns:
        - returns the Tensors that are edited ready for loss computation.
    """
    with tf.name_scope('prepare_data_for_loss'):
        # Flatten the tensors:
        if len(y_pred.shape) > (len(y_true.shape) - 1):
            y_pred = tf.reshape(y_pred, [-1, nb_classes])
        else:
            y_pred = tf.reshape(y_pred, [-1])

        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])

        if ignore_value is not None:
            # remove all elements in the image that are in the ignored class. So to compare
            # only the valid classes. To do this, compute the mask of the valid label pixels:
            mask_for_valid_labels = get_mask_for_valid_labels(y_true=y_true,
                                                              nb_classes=nb_classes,
                                                              ignore_value=ignore_value)

            # Using this mask to remove all pixels that are not in the valid classes:
            y_true = tf.boolean_mask(tensor=y_true,
                                     mask=mask_for_valid_labels,
                                     axis=0,
                                     name='gt_valid')

            y_pred = tf.boolean_mask(tensor=y_pred,
                                     mask=mask_for_valid_labels,
                                     axis=0,
                                     name='pred_valid')

    return y_true, y_pred


def prepare_class_weight_map(y_true, weights):
    """ This prepares a Pixel Weight Map that is based on the per-class weighing.
    Parameters:
        - y_true, is the Ground-Truth label map(s), with shape = (B x H x W).
        - weights, is the 1D tensor of shape (N, ) that contains the weight value for each of the
            N number of classes.
    Retturns:
        - returns the Weight Map of shape (B x H x W)
    """
    y_true_one_hot = tf.one_hot(y_true, tf.shape(weights)[0])
    weight_map = tf.tensordot(y_true_one_hot, weights, axes=1)

    return weight_map


def prepare_outline_weight_map(y_true, nb_classes, outline_size=5, outline_val=4., default_val=1.):
    """ This will prepare the pixel weight map that are based on the Outlines of each class.
    Parameters:
        - y_true, is the Ground-Truth label maps of shape (B x H x W).
        - nb_classes, is the number of classes.
        - outline_size, is the Outline size or thickness.
        - outline_val, is the weight value for the outline pixels.
        - default_val, is the weight value for the other pixels.
    Return:
        - returns a Weight map with the shape (B x H x W)
    """
    y_true_one_hot = tf.squeeze(tf.one_hot(y_true, nb_classes), axis=-2)  # shape=(3, 512, 512, 19)

    # Convert from float32 to int32: for compatibility with "binary_outline" func.
    #     y_true_one_hot = tf.cast(x=y_true_one_hot, dtype=tf.float32)

    outline_map_perClass = binary_outline(x=y_true_one_hot,
                                          kernel_size=outline_size)  # accepts shape=(3, 512, 512, 19)

    outline_map = tf.reduce_max(input_tensor=outline_map_perClass,
                                axis=-1)

    outline_map = outline_map * (outline_val - default_val) + default_val

    return outline_map



#=========================== Binary Operations Definitions ===========================

def log_n(x, n=10):
    """ This computes the log_n(x) function, where log base is `n` value of `x`.
    Parameters:
        - x, is the Input Tensor.
        - n, is the log base value.
    Returns:
        - returns the computed log result.
    """
    log_e = tf.math.log(x)
    div_log_n = tf.math.log(tf.constant(n, dtype=log_e.dtype))
    return log_e / div_log_n


def binary_dilation(x, kernel_size=3):
    """ This builds the function to perform Dilation.
        It applies dilation of a binary tensor where each of the
        input channels are processed independently.
    Parameters:
        - x, is the Binary Tensor with shape of (B x H x W x C)
        - kernel_size, is the kernel size.
    Returns:
        - returns a Dilated Tensor.
    """
    with tf.name_scope("binary_dilation"):
        nb_channels = tf.shape(x)[-1]
        kernel = tf.ones((kernel_size, kernel_size, nb_channels, 1), dtype=x.dtype)
        conv = tf.nn.depthwise_conv2d(input=x,
                                      filter=kernel,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
        clip = tf.clip_by_value(t=conv,
                                clip_value_min=1.,
                                clip_value_max=2.) - 1.

        return clip


def binary_erosion(x, kernel_size=3):
    """ This builds the function to perform Erosion.
        It applies erosion of a binary tensor where each of the
        input channels are processed independently.
    Parameters:
        - x, is the Binary Tensor with shape of (B x H x W x C) or (H x W x C) for a single image
        - kernel_size, is the kernel size.
    Returns:
        - returns a Eroded Tensor.
    """
    with tf.name_scope("binary_erosion"):
        nb_channels = tf.shape(x)[-1]
        kernel = tf.ones((kernel_size, kernel_size, nb_channels, 1), dtype=x.dtype)
        conv = tf.nn.depthwise_conv2d(input=x,
                                      filter=kernel,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
        max_val = tf.constant(kernel_size * kernel_size, dtype=x.dtype)
        clip = tf.clip_by_value(t=conv,
                                clip_value_min=max_val - 1.,
                                clip_value_max=max_val)

        return clip - (max_val - 1)


def binary_opening(x, kernel_size=3):
    """ This builds the function to perform Opening (Erosion first then dilation).
        It applies Opening of a binary tensor where each of the
        input channels are processed independently.
    Parameters:
        - x, is the Binary Tensor with shape of (B x H x W x C) or (H x W x C) for a single image
        - kernel_size, is the kernel size.
    Returns:
        - returns a Opened (Erosion first then dilation) Tensor.
    """
    with tf.name_scope("binary_opening"):
        return binary_dilation(x=binary_erosion(x, kernel_size),
                               kernel_size=kernel_size)


def binary_closing(x, kernel_size=3):
    """ This builds the function to perform Closing (Dilation first then erosion).
        It applies Closing of a binary tensor where each of the
        input channels are processed independently.
    Parameters:
        - x, is the Binary Tensor with shape of (B x H x W x C) or (H x W x C) for a single image
        - kernel_size, is the kernel size.
    Returns:
        - returns a Closed (Dilation first then erosion) Tensor.
    """
    with tf.name_scope("binary_closing"):
        return binary_erosion(x=binary_dilation(x, kernel_size),
                              kernel_size=kernel_size)


def binary_outline(x, kernel_size=3):
    """ This builds the function to perform Outline extraction (cf. cv2.morphologyEx).
        It applies Outline extraction of a binary tensor where each of the
        input channels are processed independently.
    Parameters:
        - x, is the Binary Tensor with shape of (B x H x W x C) or (H x W x C) for a single image
        - kernel_size, is the kernel size.
    Returns:
        - returns a Outline extracted Tensor.
    """
    with tf.name_scope("binary_outline"):
        return binary_dilation(x=x,
                               kernel_size=kernel_size) - binary_erosion(x=x,
                                                                         kernel_size=kernel_size)





