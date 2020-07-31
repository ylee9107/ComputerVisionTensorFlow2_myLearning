"""
File name: DataPrepCIFAR_utility.py
Author: YSLee
Date created: 01.06.2020
Date last modified: 01.06.2020
Python Version: "3.7"

"""

#=========================== Import the Libraries ===========================

import tensorflow as tf
import tensorflow_datasets as tfds
import functools

#=========================== Selecting the Dataset  ===========================
# Select the data-set to download:
dataset = "cifar100"
CIFAR_BUILDER = tfds.builder(dataset)
CIFAR_BUILDER.download_and_prepare()

#=========================== Data Preparation Functions ===========================
def _prepare_data_func(features, input_shape, augment=False, return_batch_as_tuple=True, seed=None):
    """ This builds a pre-processing function to resize the image into the expected dimensions and allows for an
        optional transformation of the images such as augmentations.
    Parameters:
        - features, is the input data.
        - input_shape, is the expected shape for model by resizing.
        - augment, is a Flag to apply random transformations to the input images.
    Returns:
        - returns Augmented images, Labels
    """

    # Convert the input images into tensors:
    input_shape = tf.convert_to_tensor(input_shape)

    # To train Keras models, it is more better(recommended) to return the batch content as tuples.
    # Convert TF-dataset feature dictionaries into Tuples:
    image = features['image']

    # Convert the image data type to "float32" and normalise the data to a range between "0 and 1":
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Data Image Augmentation:
    if augment:
        # Apply random horizontal flip::
        image = tf.image.random_flip_left_right(image, seed=seed)

        # Apply Brightness and Saturation changes:
        image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.clip_by_value(image, 0.0, 1.0)  # this keeps the pixel values in check.

        # Apply random resizing and cropping back to expected image sizes:
        random_scale_factor = tf.random.uniform([1], minval=1., maxval=1.4, dtype=tf.float32, seed=seed)

        scaled_height = tf.cast(tf.cast(input_shape[0], tf.float32) * random_scale_factor, tf.int32)
        scaled_width = tf.cast(tf.cast(input_shape[1], tf.float32) * random_scale_factor, tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))

        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, input_shape, seed=seed)
    else:
        image = tf.image.resize(image, input_shape[:2])

    if return_batch_as_tuple:
        label = features['label']
        features = (image, label)

    else:
        features['image'] = image

    return features


def get_info():
    """ This will return the information available on the dataset from Tensorflow-Dataset.
    Dataset:
        - Current Dataset is CIFAR-100.
    :return:
        - Dataset information.
    """
    return CIFAR_BUILDER.info


def get_dataset(phase='train', batch_size=32, nb_epochs=None, shuffle=True, input_shape=(32, 32, 3), return_batch_as_tuple=True, seed=None):
    """ This builds a function to process and get the dataset. Dataset is CIFAR-100.
     Parameters:
         - phase, is the current phase of data processing, either 'train' or 'test'.
         - batch_size, is the batch_size.
         - nb_epochs, is the number of epochs.
         - shuffle, is a FLag to shuffle the dataset (default=True).
         - input_shape, is the shape of the processed images.
         - return_batch_as_tuple, is a Flag to return the batched data as a tuple rather than a dict.
         - seed, is the seed number for random operations, allows for reproducibility.
    :return:
        - returns an Iterable Dataset.
    Notes:
        - Tensorflow-Dataset returns batches as feature dictionaries, that are expected by Estimators.
        - To train for Keras models, it is better to return the batch content as tuples.
    """

    # Detect Early Problems with dataset:
    assert (phase == 'train' or phase == 'test')
    is_train = phase = 'train'

    # Instantiate the data preparation function:
    prep_dat_func = functools.partial(_prepare_data_func, return_batch_as_tuple=return_batch_as_tuple,
                                      input_shape=input_shape, augment=is_train, seed=seed)

    # Take the data as Train or Test:
    cifar_data = CIFAR_BUILDER.as_dataset(split=tfds.Split.TRAIN if phase == 'train' else tfds.Split.TEST)
    cifar_data = cifar_data.repeat(nb_epochs)

    # Data Shuffling:
    if shuffle:
        cifar_data = cifar_data.shuffle(10000, seed=seed)

    # Apply the data prep func to the dataset:
    cifar_data = cifar_data.map(prep_dat_func, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    # Split the dataset into batched images:
    cifar_data = cifar_data.batch(batch_size)

    # Set to prefetch the data: for improved performance.
    cifar_data = cifar_data.prefetch(1)

    return cifar_data






