"""
File name: YOLOv3_Utilities.py
Author: YSLee
Date created: 16.06.2020
Date last modified: 16.06.2020
Python Version: "3.7"
"""

#=========================== Import the Libraries ===========================
import cv2
import numpy as np
import tensorflow as tf
from absl import logging
from itertools import repeat
from tensorflow.keras import Model
from tensorflow.keras.layers import (Add, Concatenate, Lambda,
                                     Conv2D, Input, LeakyReLU,
                                     MaxPool2D, UpSampling2D, ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (binary_crossentropy,
                                     sparse_categorical_crossentropy)
from YOLOv3_Utilities import (interval_overlap, intersectionOverUnion)

#=========================== Anchor Boxes ===========================
# Define the Yolo Anchor boxes:
yolo_anchors = np.array( [(10, 13), (16, 30), (33, 23),
                          (30, 61), (62, 45), (59, 119),
                          (116, 90), (156, 198), (373, 326)], np.float32 ) / 416

# Define the Yolo Anchor Masks:
yolo_anchor_masks = np.array( [[6, 7, 8], [3, 4, 5], [0, 1, 2]] )

#=========================== IoU Variables ===========================

# Define the Threshold for YOLO:
yolo_iou_threshold = 0.6

# Define the Threshold Score:
yolo_score_threshold = 0.6

#=========================== Yolo v3 Model Functions ===========================

class BatchNormalisation(tf.keras.layers.BatchNormalization):
    """ This builds the Batch Normalisation function for YOLOv3 to be compatible with Transfer Learning Methods.
        It also inherits the properties from tf.keras.layers.BatchNormalization.

    Parameters: within the "call" method.
        - Make trainable = False freeze BN for real.
    """

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)

        training = tf.logical_and(training, self.trainable)

        return super().call(x, training)


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    """ This builds the Convolutions Filters over the input image.
    Parameters:
        - x, is the input image tensor.
        - filters, is the number of filters/kernels.
        - size, is the size of the kernels.
        - strides, is the convoluitional stride.
        - batch_norm, is a Flas to use or not the batch normalisation in this convolutional layer.
    Returns:
        - returns the processed (convolved) x output.
    """
    # Set the padding to ouput the same dimension/shape as the input if stride = 1:
    if strides == 1:
        padding = 'same'
    else:
        # Padding applied to only the top left of the input tensor:
        x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
        padding = 'valid'
    x = Conv2D(filters=filters,
               kernel_size=size,
               strides=strides,
               padding=padding,
               use_bias=not batch_norm,
               kernel_regularizer=l2(l=0.0005)
               )(x)

    # If this layer uses batch_norm, set the following:
    if batch_norm:
        x = BatchNormalisation()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return x


def DarknetResidual(x, filters):
    """ This builds the Residual block.
    Parameters:
        - x, is the input tensor.
        - filters, is the number of filters/kernels
    Returns:
        - returns the processed x output.
    """
    # Save an instance of x to be used for merging later:
    previous_x = x

    # 1st DarknetConv_BatchNorm_LeakyReLU (DBL) block:
    x = DarknetConv(x=x,
                    filters=filters // 2,
                    size=1)
    # 2nd DarknetConv_BatchNorm_LeakyReLU (DBL) block:
    x = DarknetConv(x=x,
                    filters=filters,
                    size=3)

    # Merge the Residual and DBL:
    x = Add()([previous_x, x])

    return x


def DarknetBlock(x, filters, blocks):
    """ This builds the Resblock_body block.
    Parameters:
        - x, is the input tensor.
        - filters, is the number of kernels/filters.
        - blocks, is the number of repeated blocks to be set.
    Returns:
        - returns the processed x output of this ResN block.
    """
    # 1st block with zero padding and DarknetConv_BatchNorm_LeakyReLU (DBL) block:
    x = DarknetConv(x=x,
                    filters=filters,
                    size=3,
                    strides=2)

    # Repeated ResN blocks (or Redisdual blocks):
    for _ in repeat(None, blocks):
        x = DarknetResidual(x=x,
                            filters=filters)

    return x


def Darknet(name=None):
    """ This builds the 53-Layer Darknet model.
    Parameters:
        - name, is the Name Suffix of this layer.
    Returns:
        - returns the model.
    """
    # Input:
    x = inputs = Input([None, None, 3])

    # 1st layer after input: DarknetConv_BatchNorm_LeakyReLU (DBL) block.
    x = DarknetConv(x=x, filters=32, size=3)

    # 2nd Block: res1 block.
    x = DarknetBlock(x=x, filters=64, blocks=1)

    # 3rd Block: res2 block.
    x = DarknetBlock(x=x, filters=128, blocks=2)

    # 4th Block: res8 block.
    x = x_36 = DarknetBlock(x=x, filters=256, blocks=8)

    # 5th Block: res8 block.
    x = x_61 = DarknetBlock(x=x, filters=512, blocks=8)

    # 6th Block: res4 block.
    x = DarknetBlock(x=x, filters=1024, blocks=4)

    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def YoloConv(filters, name=None):
    """ This builds the Yolo Convolutions (DBL * 5) and the (DBL + Up-Sampling).
    Parameters:
        - filters, is the number of filters.
        - name, is the Name Suffix of this layer.
    Returns:
        - returns a callable yolo_conv layer.
    """

    def yolo_conv(x_in):

        # For DBL with Up-sampling:
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])

            x, x_skip = inputs

            x = DarknetConv(x=x, filters=filters, size=1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])

        else:
            x = inputs = Input(x_in.shape[1:])

        # For DBL * 5 Layer:
        x = DarknetConv(x=x, filters=filters, size=1)
        x = DarknetConv(x=x, filters=filters * 2, size=3)
        x = DarknetConv(x=x, filters=filters, size=1)
        x = DarknetConv(x=x, filters=filters * 2, size=3)
        x = DarknetConv(x=x, filters=filters, size=1)

        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    """ This builds the Yolo Output blocks, where it can be configured for different channels.
    Parameters:
        - filters, is the number of filters/kernels.
        - anchors, is the defined number of anchor boxes (masks).
        - classes, is the number of classes.
        - name, is the Name Suffix of this layer.
    Returns:
        - returns a callable yolo output layer.
    """

    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x=x,
                        filters=filters * 2,
                        size=3)
        x = DarknetConv(x=x,
                        filters=anchors * (classes + 5),
                        size=1,
                        batch_norm=False)
        x = Lambda(lambda x: tf.reshape(tensor=x,
                                        shape=(-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)
                                        )
                   )(x)

        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


def yolo_boxes(pred, anchors, classes):
    """ This builds the YOLO model output prediction boxes.
    Parameters:
        - pred, is the input predictions from the model.
        - anchors, is the defined number of anchor boxes (masks).
        - classes, is the number of classes.
    Returns:
        - returns the bounding boxes (bbox), score, class_probs, and pred_box.
    """
    # Define the Grid Size:
    grid_size = tf.shape(pred)[1]

    # Define the box x-y, w-h, score and class_prob:
    box_xy, box_wh, score, class_probs = tf.split(value=pred,
                                                  num_or_size_splits=(2, 2, 1, classes),
                                                  axis=-1)

    # Refine the Acnhor boxes (part 1):
    box_xy = tf.sigmoid(box_xy)
    score = tf.sigmoid(score)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    # Build the Grid:
    grid = tf.meshgrid(tf.range(grid_size),
                       tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1),
                          axis=2)

    # Refine the Anchor Boxes (part 2):
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2

    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, score, class_probs, pred_box


def nonMaximumSuppression(outputs, anchors, masks, classes):
    """ This builds the NMS, where it remove all the other unwanted bounding boxes (with the lowest probability).
    Parameters:
        - outputs, is the output tensor from the previous layer.
        - anchors, is the number of anchors.
        - masks, is the anchor masks.
        - classes, is the number of classes.
    Returns:
        - returns boxes, scores, classes, valid_detections.
    """
    # Define empty arrays:
    boxes, conf, out_type = [], [], []

    # Append the empty arrays:
    for output in outputs:
        boxes.append(tf.reshape(tensor=output[0],
                                shape=(tf.shape(output[0])[0], -1, tf.shape(output[0])[-1])
                                )
                     )
        conf.append(tf.reshape(tensor=output[1],
                               shape=(tf.shape(output[1])[0], -1, tf.shape(output[1])[-1])
                               )
                    )
        out_type.append(tf.reshape(tensor=output[2],
                                   shape=(tf.shape(output[2])[0], -1, tf.shape(output[2])[-1])
                                   )
                        )

    # Define the bounding box, confidence and class_probs: axis=1 means concat to the right side, not below.
    bbox = tf.concat(values=boxes, axis=1)
    confidence = tf.concat(values=conf, axis=1)
    class_probs = tf.concat(values=out_type, axis=1)

    # Compute the scores:
    scores = confidence * class_probs

    # Apply TensorFlow NMS:
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
        )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=80, training=False):
    """ This builds the YoloV3 model, by combining all the blocks.
    Parameters:
        - size, is the kernels/filers size.
        - channels, is the number of RGB =3 colour channels.
        - anchors, is the yolo anchors.
        - masks, is the yolo anchors masks.
        - classes, is the number of classes.
        - training, is a Flag to state whether the model instantiation will will to be trained or not.
    Returns:
        - returns the YoloV3 model.
    """
    # Input layer:
    x = inputs = Input( [size, size, channels] )

    # Instantiate the Darknet-53 without FC layer:
    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    # Define the yolo_head blocks for each type of (512, 256, 128) or (1024, 512, 256) chanel:
    # For masks -> [6, 7, 8]
    x = YoloConv(filters=512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(filters=512, anchors=len(masks[0]), classes=classes, name='yolo_output_0')(x)

    # For masks -> [3, 4, 5]
    x = YoloConv(filters=256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(filters=256, anchors=len(masks[1]), classes=classes, name='yolo_output_1')(x)

    # For masks -> [0, 1, 2]
    x = YoloConv(filters=128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(filters=128, anchors=len(masks[2]), classes=classes, name='yolo_output_2')(x)

    # If Flag 'training' = True:
    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    # Compute the bounding boxes:
    boxes_0 = Lambda(lambda x: yolo_boxes(pred=x,
                                          anchors=anchors[masks[0]],
                                          classes=classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(pred=x,
                                          anchors=anchors[masks[1]],
                                          classes=classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(pred=x,
                                          anchors=anchors[masks[2]],
                                          classes=classes), name='yolo_boxes_2')(output_2)

    # Apply NMS on the candidate bounding boxes:
    outputs = Lambda(lambda x: nonMaximumSuppression(outputs=x,
                                                     anchors=anchors,
                                                     masks=masks,
                                                     classes=classes),
                     name='nonMaximumSuppression')( (boxes_0[:3], boxes_1[:3], boxes_2[:3]) )

    return Model(inputs, outputs, name='yolov3')


def YoloLoss(anchors, classes=80, ignore_threshold=0.5):
    """ This builds for the compute of the model losses.
    Parameters:
        - anchors, is the yolo_anchors or anchor boxes.
        - classes, is the number of classes.
        - ignore_threshold, is when not specified, the threshold will be default at 0.5.
    Returns:
        - returns yolo_loss, where = xy_loss + wh_loss + obj_loss + class_loss.
    """

    def yolo_loss(y_true, y_pred):
        # Part 1 - Transform all pred outputs:
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(pred=y_pred, anchors=anchors, classes=classes)

        # Split the predicted box shape lines:
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # Part 2 - Transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(value=y_true, num_or_size_splits=(4, 1, 1), axis=-1)

        # Split the Ground truth box shape lines:
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # Allow for higher weights to small boxes:
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # Part 3 - Inverting the pred box equations:
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

        # Convert true_xy to match grid size:
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # Part 4 - Compute all masks:
        obj_mask = tf.squeeze(input=true_obj, axis=-1)

        # Ignore when IoU is over the threshold stated:
        true_box_flat = tf.boolean_mask(tensor=true_box, mask=tf.cast(obj_mask, tf.bool))

        # Find the Best IoU:
        best_iou = tf.reduce_max(intersectionOverUnion(box1=pred_box, box2=true_box_flat), axis=-1)

        # Ignore IoU:
        ignore_mask = tf.cast(best_iou < ignore_threshold, tf.float32)

        # Part 5 - Compute all losses:
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(input_tensor=tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(input_tensor=tf.square(true_wh - pred_wh), axis=-1)

        # Using binary_crossentropy for the object loss:
        obj_loss = binary_crossentropy(y_true=true_obj, y_pred=pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

        # Using sparse_categorical_crossentropy instead for the classes loss:
        class_loss = obj_mask * sparse_categorical_crossentropy(y_true=true_class_idx, y_pred=pred_class)

        # Part 6 - Sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss


