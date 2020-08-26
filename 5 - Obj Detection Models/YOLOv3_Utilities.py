"""
File name: YOLOv3_Utilities.py
Author: YSLee
Date created: 16.06.2020
Date last modified: 16.06.2020
Python Version: "3.7"
"""

#=========================== Import the Libraries ===========================
from absl import logging
import numpy as np
import tensorflow as tf
import cv2

#=========================== List of YOLOv3 layers ===========================
YOLOV3_LAYER_LIST = ['yolo_darknet',
                     'yolo_conv_0',
                     'yolo_output_0',
                     'yolo_conv_1',
                     'yolo_output_1',
                     'yolo_conv_2',
                     'yolo_output_2']

#=========================== Dataset Utilities ===========================
@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs, classes):
    """ This builds the function to transform targets outputs tuple of shape.
        (
            [N, 13, 13, 3, 6],
            [N, 26, 26, 3, 6],
            [N, 52, 52, 3, 6]
        )

    Parameters:
        - y_true, is the labels.
        - grid_size, is the size of the grid.
        - anchor_idxs, is the anchor indexes.
        - classes, is the number of classes.
    Returns:
        - returns an updated scattered tensor.

    """
    # For y_true, shape is (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # For y_true_out, shape is (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    # Define array for indexes and updates:
    indexes = tf.TensorArray(tf.int32, size=1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

    # Loop over labels to get anchor boxes:
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_mask, classes):
    """ This transforms the Outputs -> [x, y, w, h, object, class].
    Parameters:
        - y_train, are the labels from the training set.
        - anchors, are the anchor boxes.
        - anchor_masks, are the anchor boxes masks.
        - classes, is the number of classses.
    Returns:
        - returns a tuple of the output -> [x, y, w, h, obj, class]
    """
    # Define the array for the outputs:
    outputs = []

    # Define the Grid size:
    grid_size = 13

    # Compute the anchor index for true boxes:
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]

    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(input=tf.expand_dims(input=box_wh, axis=-2),
                     multiples=(1, 1, tf.shape(anchors)[0], 1)
                     )

    box_area = box_wh[..., 0] * box_wh[..., 1]

    intersection = tf.minimum(x=box_wh[..., 0], y=anchors[..., 0]) * tf.minimum(x=box_wh[..., 1], y=anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)

    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(input=anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_i in anchor_masks:
        outputs.append(transform_targets_for_output(y_true=y_train,
                                                    grid_size=grid_size,
                                                    anchor_idxs=anchor_i,
                                                    classes=classes))
        grid_size *= 2

    return tuple(outputs)

def preprocess_image(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

#=========================== Model Utilities ===========================
def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)

        for i, layer in enumerate(sub_model.layers):

            # If the layer name does not start with 'conv2d', skip it and move to the next layer:
            # As this is the convolutional layer, there is no batch_norm set.
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None

            # For all the layers that consist of Batch Norm within the model, if the layer starts with "batch_norm"
            # set it as the batch_norm variable:
            if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            # Log the layers layout for each sub_model:
            logging.info("{}/{} {}".format(sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            # Define the filters:
            filters = layer.filters

            # Define the filter size:
            size = layer.kernel_size[0]

            # Define the input dimensions:
            input_dim = layer.input_shape[-1]

            # When batch_norm is None, set it to convolutional bias:
            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]:
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)

                # tf [gamma, beta, mean, variance]:
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width):
            conv_shape = (filters, input_dim, size, size)

            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))

            # tf shape (height, width, in_dim, out_dim):
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            # Set the weights for the Convolutional layers and the bias:
            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                # Set the weights for the batch norm:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read the weights.'

    wf.close()


def interval_overlap(interval_1, interval_2):
    x1, x2 = interval_1
    x3, x4 = interval_2

    if x3 < x1:
        return 0 if x4 < x1 else (min(x2, x4) - x1)
    else:
        return 0 if x2 < x3 else (mi(x2, x4) - x3)


def intersectionOverUnion(box1, box2):
    # Compute the intersect width and height:
    intersect_width = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_height = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    # Compute the intersection area:
    intersect_area = intersect_width * intersect_height

    # Get the length of height and width for both boxes:
    w1 = box1.xmax - box1.xmin
    h1 = box1.ymax - box1.ymin

    w2 = box2.xmax - box2.xmin
    h2 = box2.ymax - box2.ymin

    # Compute the Area of Union:
    union_area = (w1 * h1) + (w2 * h2) - intersect_area

    return float(intersect_area) / union_area


def draw_outputs(img, outputs, class_names):
    boxes, score, classes, nums = outputs
    boxes, score, classes, nums = boxes[0], score[0], classes[0], nums[0]

    wh = np.flip(img.shape[0:2])

    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)

        img = cv2.putText(img=img,
                          text='{} {:.3f}'.format(class_names[int(classes[i])], score[i]),
                          org=x1y1,
                          fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          fontScale=1,
                          color=(0, 0, 255),
                          thickness=2)

    return img




