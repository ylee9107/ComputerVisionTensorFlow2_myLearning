"""
File name: cdataset_utilities.py
Author: YSLee
Date created: 20.07.2020
Date last modified:20.07.2020
Python Version: "3.7"
"""

#=========================== Import the Libraries ===========================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import cityscapesscripts.helpers.labels as cityscapes_labels
import glob
import numpy as np
import functools

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

#=========================== Defined Functions/Methods ===========================
#=========================== Part 1 - DATA FUNCTIONS =============================

def extract_cityscapes_file_pairs(split='train', city="*", sequence='*', frame='*', ext='.*',
                                  gt_type='labelTrainIds', type='leftImg8bit',
                                  root_folder=CITYSCAPES_FOLDER, file_template=CITYSCAPES_FILE_TEMPLATE):
    """ This builds a function to Extract the filenames for the CityScapes Dataset
        Note: to account for wildcards in the parameters, city='*' is set, to return
              image paris from every city.
        Parameters:
            - split, is the name of the split to return pairs from ("train" or "val").
            - city, is the name of the City or Cities.
            - sequence, is the name of the video sequence(s).
            - frame, is the frame.
            - ext, is the extension.
            - gt_type, is the Cityscapes GT type.
            - type, is ht eCityscapes image type.
            - root_folder, is the Cityscapes root folder.
            - file_template, is the file template to be applied (where default = corresponds to Cityscapes original format)
        Returns:
            - returns a List of input files, List of corresponding GT files.
    """
    input_file_template = file_template.format(root=root_folder, type=type, type2='',
                                               len_fill=1, filler='*', split=split, city=city,
                                               seq=sequence, frame=frame, ext=ext)
    input_files = glob.glob(input_file_template)

    gt_file_template = file_template.format(root=root_folder, type='gtFine', type2='_' + gt_type,
                                            len_fill=1, filler='*', split=split, city=city,
                                            seq=sequence, frame=frame, ext=ext)
    gt_files = glob.glob(gt_file_template)

    assert (len(input_files) == len(gt_files))
    return sorted(input_files), sorted(gt_files)


def parse_func(filenames, resize_to=[226, 226], augment=True):
    """ This Parses the files into input/label image pair.
    Parameters:
        - filenames, is the Dict that contains the file(s). (filenames['image'], filenames['label'])
        - resize_to, is the Height x Width Dimensions to resize the image and label into.
        - augment, is an optional Flag to augment the data pairs.
    Returns:
        - returns an Input tensor, Label tensor.
    """
    # Get the image and label filenames:
    img_filename, gt_filename = filenames['image'], filenames.get('label', None)

    # Read the file and return as Bytes:
    image_string = tf.io.read_file(img_filename)

    # Decode the data into an Image:
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)

    # Convert the image to Float:
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    # Resize the image:
    image = tf.image.resize(image, resize_to)

    # Apply the same as above to the Labels:
    if gt_filename is not None:
        gt_string = tf.io.read_file(gt_filename)
        gt_decoded = tf.io.decode_png(gt_string, channels=1)
        gt = tf.cast(gt_decoded, dtype=tf.int32)
        gt = tf.image.resize(gt, resize_to, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Augmenting: if True
        if augment:
            image, gt = _augmenation_func(image, gt)
        return image, gt

    else:
        if augment:
            image = _augmenation_func(image)
        return image


def _augmenation_func(image, gt_image=None):
    """ This builds a function to apply random transforamtion to augment the training images.
    Parameters:
        - images, is the input images.
    Returns:
        - returns Augmented Images.
    Note:
        To randomly flip or crop the images, will require the same to be applied to the label for
        consistency.
    """
    original_shape = tf.shape(image)[-3:-1]
    nb_image_chnls = tf.shape(image)[-1]

    # Stack the image and label together along the channel axis for the random operations
    # (flip or resize/crop) to be applied to both:
    if gt_image is None:
        stacked_images = image
        nb_stacked_chnls = nb_image_chnls
    else:
        stacked_images = tf.concat([image, tf.cast(gt_image, dtype=image.dtype)], axis=-1)
        nb_stacked_chnls = tf.shape(stacked_images)[-1]

    # Apply random horizontal flip:
    stacked_images = tf.image.random_flip_left_right(stacked_images)

    # Apply random cropping:
    random_scale_factor = tf.random.uniform([], minval=.8, maxval=1., dtype=tf.float32)
    crop_shape = tf.cast(tf.cast(original_shape, tf.float32) * random_scale_factor, tf.int32)

    if len(stacked_images.shape) == 3:
        # For a single images:
        crop_shape = tf.concat([crop_shape, [nb_stacked_chnls]], axis=0)
    else:
        # for batched images:
        batch_size = tf.shape(stacked_images)[0]
        crop_shape = tf.concat([[batch_size], crop_shape, [nb_stacked_chnls]], axis=0)
    stacked_images = tf.image.random_crop(stacked_images, crop_shape)

    # The following transformations will be applied differently to the input and gt images
    # (nearest-neighbor resizing for the label image VS interpolated resizing for the image),
    # or applied only to the input image. Hence, split them back:
    image = stacked_images[..., :nb_image_chnls]

    # Resize the image back to the expected dimensions:
    image = tf.image.resize(image, original_shape)

    # Apply random Brightness/Saturation changes:
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.75)
    image = tf.clip_by_value(image, 0.0, 1.0)  # to keep pixel values in check.

    if gt_image is not None:
        # gt_image = tf.cast(stacked_images[..., nb_image_chnls:], dtype=gt_image.dtype)
        gt_image = tf.cast(stacked_images[..., nb_image_chnls:], dtype=tf.float32)
        gt_image = tf.image.resize(gt_image, original_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image, gt_image
    else:
        return image

def segmentation_input_func(image_files, gt_files=None, resize=[256, 256], shuffle=False,
                            batch_size=32, nb_epochs=None, augment=False, seed=None):
    """ This is the Input Data Pipeline for Semantic Segmentation applications.
    Parmeters:
        - image_files, is the List of input image files.
        - gt_files, is an optional List of corresponding label image files.
        - resize_to, is the Heigh x Width dimensions for resizing the image and label.
        - shuffle, is a Flag to shuffle the dataset.
        - batch_size, is the batch size.
        - nb_epochs, is the number of epochs the dataset will be iterated over.
        - augment, is an optional Flag to augment the image pairs.
        - seed, is an optional Flag to set the seed for reproducibility purposes.
    Returns:
        - returns tf.data.Dataset

    """
    # Converting to TensorFlow Dataset format:
    image_files = tf.constant(image_files)
    data_dict = {'image': image_files}

    if gt_files is not None:
        gt_files = tf.constant(gt_files)
        data_dict['label'] = gt_files

    # Get the slices of an array in the form of objects (Dict):
    dataset = tf.data.Dataset.from_tensor_slices(data_dict)

    # If Shuffle is set to True:
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    dataset = dataset.prefetch(1)

    # Batching and Adding the Parsing Operation:
    parse_fn = functools.partial(parse_func, resize_to=resize, augment=augment)
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(nb_epochs)

    return dataset

def cityscapes_input_func(split='train', root_folder=CITYSCAPES_FOLDER, resize_to=[256, 256], shuffle=False,
                          batch_size=32, nb_epochs=None, augment=False, seed=None, blurred=False):
    """ This sets up the Input Data Pipeline for Semantic Segmentation applications for
        the Dataset -> Cityscapes dataset.
    Parameters:
        - split, is the Split name such as ('train', 'val', 'test')
        - root_folder, is the Cityscapes root folder.
        - resize_to, is the Height x Width dimensions to resize the image and label.
        - shuffle, is an optional Flag to shuffle the dataset.
        - batch_size, is the Batch size.
        - nb_epochs, is the number of epochs that the dataset would be iterated over.
        - augment, is the Flag to augment the image pairs.
        - seed, is an optional Flag to set the seed for reproducibility purposes.
        - blurred, is the FLag to use images with faces and immatriculation plates blurred
                   (for display).
    Returns:
        - returns tf.data.Dataset
    """
    # Set up the dataset type: blurred or not.
    type = "leftImg8bit_blurred" if blurred else "leftImg8bit"

    # Process the files:
    input_files, gt_files = extract_cityscapes_file_pairs(split=split,
                                                          type=type,
                                                          root_folder=root_folder)
    dataset = segmentation_input_func(image_files=input_files,
                                      gt_files=gt_files,
                                      resize=resize_to,
                                      shuffle=shuffle,
                                      batch_size=batch_size,
                                      nb_epochs=nb_epochs,
                                      augment=augment,
                                      seed=seed)

    return dataset


# def cityscapes_input_func(split='train', root_folder=CITYSCAPES_FOLDER, resize_to=[256, 256], shuffle=False,
#                           batch_size=32, nb_epochs=None, augment=False, seed=None, blurred=False):
#     """ This sets up the Input Data Pipeline for Semantic Segmentation applications for
#         the Dataset -> Cityscapes dataset.
#     Parameters:
#         - split, is the Split name such as ('train', 'val', 'test')
#         - root_folder, is the Cityscapes root folder.
#         - resize_to, is the Height x Width dimensions to resize the image and label.
#         - shuffle, is an optional Flag to shuffle the dataset.
#         - batch_size, is the Batch size.
#         - nb_epochs, is the number of epochs that the dataset would be iterated over.
#         - augment, is the Flag to augment the image pairs.
#         - seed, is an optional Flag to set the seed for reproducibility purposes.
#         - blurred, is the FLag to use images with faces and immatriculation plates blurred
#                    (for display).
#     Returns:
#         - returns tf.data.Dataset
#     """
#     # Set up the dataset type: blurred or not.
#     type = "leftImg8bit_blurred" if blurred else "leftImg8bit"
#
#     # Process the files:
#     input_files, gt_files = extract_cityscapes_file_pairs(split=split,
#                                                           type=type,
#                                                           root_folder=root_folder)
#
#     # Convert to Float32:
#
#     return segmentation_input_func(image_files=input_files,
#                                    gt_files=gt_files,
#                                    resize=resize_to,
#                                    shuffle=shuffle,
#                                    batch_size=batch_size,
#                                    nb_epochs=nb_epochs,
#                                    augment=augment,
#                                    seed=seed)

#=========================== Part 2 - DISPLAY FUNCTIONS ===========================

def change_ratio(image=None, pred=None, gt=None, ratio=CITYSCAPES_IMG_RATIO):
    """ This builds a function to Resize the Images to the appropriate defined ratios.
    Parameters:
        - image, is an optional Input Image.
        - pred, is an optional Predicted label image.
        - gt, is an optional Target Image.
        - ratio, is the defined ratio to be set for the images.
    Returns:
        - returns the 3 resized images.
    """
    # Check and set the input:
    valid_input = image if image is not None else pred if pred is not None else gt

    # Define the current size of the input:
    current_size = tf.shape(valid_input)[-3:-1]

    # Define the width * ratio:
    width_with_ratio = tf.cast(tf.cast(current_size[1], tf.float32) * ratio, tf.int32)

    # stack the size with the width_with_ratio:
    size_with_ratio = tf.stack([current_size[0], width_with_ratio], axis=0)

    if image is not None:
        image = tf.image.resize(images=image,
                                size=size_with_ratio)

    if pred is not None:
        pred = tf.image.resize(images=pred,
                               size=size_with_ratio,
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if gt is not None:
        gt = tf.image.resize(images=gt,
                             size=size_with_ratio,
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, pred, gt


def convert_label_to_colours(label, one_hot=True, nb_classes=len(CITYSCAPES_LABELS),
                             colour_tensor=CITYSCAPES_COLOURS_TF):
    """ This converts the Label images into Coloured ones, for the purposes of displaying thm.
    Paramters:
        - label, is the Label Image (tensor).
        - one_hot, is the Flag to one-hot encode the label images if they are not encoded.
        - nb_classes, is the number of classes for one-hot-encoding.
        - colour_tensor, is the TTensor mapping labels to colours.
    Returns:
        - returns the Colour Map.
    """
    # Define the shape and channels:
    label_shape = tf.shape(label)
    colour_channels = tf.shape(colour_tensor)[-1]

    # Check to one-hot encode the label:
    if one_hot:
        label = tf.one_hot(label, nb_classes)
    else:
        label_shape = label_shape[:-1]

    # Return the labels as coloured ones:
    label = tf.reshape(tf.cast(label, tf.int32), (-1, nb_classes))
    colours = tf.matmul(label, colour_tensor)

    return tf.reshape(colours, tf.concat([label_shape, [colour_channels]], axis=0))


def postprocess_to_show_images(image=None, pred=None, gt=None, one_hot=True, ratio=CITYSCAPES_IMG_RATIO):
    """ This post-processes the training results of the segmentation model for display.
        The training results should be tensors.
    Parameters:
        - image, is an optional Input image tensor.
        - pred, is an optional Predicted label map tensor.
        - gt, is an optional target label map tensor.
        - one_hot, is the Flag to one-hot-encode the label images if they were not done yet.
        - ratio, is the Original image ratios.
    Returns:
        - returns the processed image tensor(s).
    """
    # Define a list to store the output:
    out = []

    # Apply a change of ratios to the images:
    image_show, pred_show, gt_show = change_ratio(image, pred, gt, ratio)

    # Update the Output list:
    if image is not None:
        out.append(image_show)

    if pred is not None:
        if one_hot:
            # Remove the unnecessary channel dimensions:
            pred_show = tf.squeeze(pred_show, -1)
        pred_show = convert_label_to_colours(pred_show, one_hot=one_hot)
        out.append(pred_show)

    if gt is not None:
        # Remove the unnecessary channel dimensions:
        gt_show = tf.squeeze(gt_show, -1)
        gt_show = convert_label_to_colours(gt_show)
        out.append(gt_show)

    return out if len(out) > 1 else out[0]


def convert_labels_to_colours_numpy(label, one_hot=True, nb_classes=len(CITYSCAPES_LABELS),
                                    colour_array=CITYSCAPES_COLOURS, ignore_value=value_to_ignore):
    """ This Covnerts the Label Images into coloured ones for display. These will be Numpy Objects.
    Parameters:
        - label, is the Label image (in Numpy Arrays).
        - one_hot, is an optional Flag to one-hot-encode the label image, if they weren't beforehand.
        - nb_classes, is the number of classses for one-hot-encoding.
        - colour_array, is the Array mapping the labels to colours.
        - ignore_value, is the value of the label to be ignored (value_to_ignore) for one-hot-encoding.
    Returns:
        - returns a Colour Map.
    """
    if one_hot:
        label_shape = label.shape
        label = label.reshape(-1)

        label[label == ignore_value] = nb_classes
        label = np.eye(N=nb_classes + 1,
                       dtype=np.int32)[label]

        label = label[..., :nb_classes]

    else:
        label_shape = label.shape[:-1]
        label = label.reshape(-1, label.shape[-1])

    colours = np.matmul(label, colour_array)

    return colours.reshape(list(label_shape) + [colours.shape[1]])



















