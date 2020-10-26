"""
File name: unet_model_v2.py
Author: YSLee
Date created: 30.07.2020
Date last modified:30.07.2020
Python Version: "3.7"
"""
#=========================== Import the Libraries ===========================

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D,
                                     Conv2DTranspose,
                                     Lambda,
                                     Dropout,
                                     MaxPooling2D,
                                     LeakyReLU,
                                     concatenate,
                                     BatchNormalization)

#=========================== Helper Scaling Function Definitions ===========================
# The dimensions of the input images might not be normalised, this means that after downsmapling and
# upsampling, the original size of the image mightno be re-obtained. (there could be a difference
# of +/- 1 pixel of difference). The following will help rescale the images.

# 'images' here is a tuple of 2 tensors,
# where the 1st image tensor will be resized to the shape of the 2nd tensor.

ResizeBack = lambda name: Lambda(function = lambda images: tf.image.resize(images[0], tf.shape(images[1])[-3:-1]),
                                 name = name)

Upscale = lambda name: Lambda(function = lambda images: tf.image.resize(images, tf.shape(images)[-3:-1] * scale_factor),
                              name = name)
#=========================== Function Definitions ===========================

def unet_conv_block(x, filters, kernel_size=3, batch_norm=True,
                    dropout=False, name_prefix='enc_', name_suffix=0):
    """ This builds the U-Net Convolution Block, Pass an input tensor through 2x Convolutional layers,
        by using the convolution sequence.
        Conv Block sequence: 𝐶𝑜𝑛𝑣𝑜𝑙𝑢𝑡𝑖𝑜𝑛𝑠+(𝑜𝑝𝑡𝑖𝑜𝑛𝑎𝑙)𝐵𝑎𝑡𝑐ℎ𝑁𝑜𝑟𝑚𝑎𝑙𝑖𝑠𝑎𝑡𝑖𝑜𝑛+𝐿𝑒𝑎𝑘𝑦𝑅𝑒𝐿𝑈.
    Parameters:
        - x, is the Input Tensor.
        - filters, is the number of filters for the convolution.
        - kernel_size, is the kernel/filter size for the convolution.
        - batch_norm, is an optional Flag to perform batch normalisation.
        - dropout, is an optional Flag to perform dropout between the two convolutions.
        - name_prefix, is the prefix for the layer's name.
        - name_suffix, is the suffix for the layer's name.
    Returns:
        - returns the Transformed Tensor.
    """

    name_fn = lambda layer, num: '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)

    # 1st Conv block:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation=None,
               kernel_initializer='he_normal',
               padding='same',
               name=name_fn('conv', 1)
               )(x)

    # Apply batch norm if true:
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 1))(x)

    # Apply Activation func:
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 1))(x)

    # Apply Dropout if true:
    if dropout:
        x = Dropout(rate=0.2, name=name_fn('drop', 1))(x)

    # 2nd Conv block:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation=None,
               kernel_initializer='he_normal',
               padding='same',
               name=name_fn('conv', 2)
               )(x)

    # Apply batch norm if true:
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 2))(x)

    # Apply Activation func:
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 2))(x)

    return x


def unet_deconv_block(x, filters, kernel_size=2, strides=2, batch_norm=True,
                    dropout=False, name_prefix='dec_', name_suffix=0):
    """ This builds the U-Net De-Convolution Block. It passes the input tensor through
        1x Convolution layer first then another 1x transposed (de)Convolution layer with LeakyRelU
        with additional but optional Batch Norm. and Dropout layers.
    Parameters:
        - filters, is the number of filters for the convolution.
        - kernel_size, is the kernel/filter size for the convolution.
        - strides, is the Strides set for the transposed convolution.
        - batch_norm, is an optional Flag to perform batch normalisation.
        - dropout, is an optional Flag to perform dropout between the two convolutions.
        - name_prefix, is the prefix for the layer's name.
        - name_suffix, is the suffix for the layer's name.
    Returns:
        - returns the Transformed Tensor.
    """

    name_fn = lambda layer, num: '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)

    # 1st Conv block:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation=None,
               kernel_initializer='he_normal',
               padding='same',
               name=name_fn('conv', 1)
               )(x)

    # Apply batch norm if true:
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 1))(x)

    # Apply Activation func:
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 1))(x)

    # Apply Dropout if true:
    if dropout:
        x = Dropout(rate=0.2, name=name_fn('drop', 1))(x)

    # 2nd De-Convolution block:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        activation=None,
                        kernel_initializer='he_normal',
                        padding='same',
                        name=name_fn('conv', 2)
                        )(x)

    # Apply batch norm if true:
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 2))(x)

    # Apply Activation func:
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 2))(x)

    return x



def unet(x, out_channels=3, layer_depth=4, filters_orig=32, kernel_size=4,
         batch_norm=True,  final_activation='sigmoid'):
    """ This builds the U-Net model.
    Parameters:
        - x, is the Input Tensor.
        - out_channels, is the number of output channels.
        - layer_depth, is the number of conv blocks that are vertically stacked.
        - filters_orig, is the number of filters for the 1st CNN Layer,
            where it is then multiplied by 2 for every block.
        - kernel_size, is the size of the filter/kernel for the convolutions.
        - batch_norm, is an optional Flag to perform batch normalisation.
        - final_activation, is the name of the activation funciton used in the final layer.
    Returns:
        - returns the Output Tensor.
    """
    # Encoder Segment:
    filters = filters_orig

    outputs_for_skip = []
    for i in range(layer_depth):
        # Conv block:
        x_conv = unet_conv_block(x=x,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 batch_norm=batch_norm,
                                 name_suffix=i
                                 )

        # Save the outputs of the encoders for shortcut/skip connections:
        outputs_for_skip.append(x_conv)

        # Downsample:
        x = MaxPooling2D(pool_size=(2, 2))(x_conv)

        # Define the number of filters:
        filters = min(filters * 2, 512)

    # Bottleneck Layer:
    x = unet_conv_block(x=x,
                        filters=filters,
                        kernel_size=kernel_size,
                        name_suffix='_bottleneck'
                        )

    # Decoder Segment:
    for i in range(layer_depth):
        # Define the number of filters:
        filters = max(filters // 2, filters_orig)

        # define when to use Dropout:
        use_dropout = i < (layer_depth - 2)

        deconv_block = unet_deconv_block(x=x,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         batch_norm=batch_norm,
                                         dropout=use_dropout,
                                         name_suffix=i
                                         )

        # Shortcut:
        shortcut = outputs_for_skip[-(i + 1)]

        # Resize the image:
        deconv_block = ResizeBack(name='resize_to_same{}'.format(i))([deconv_block, shortcut])

        # Concatenate:
        x = concatenate([deconv_block, shortcut], axis=-1, name='dec_concat{}'.format(i))

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               padding='same',
               name='dec_out1'
               )(x)

    x = Dropout(rate=0.3, name='drop_out1')(x)

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               padding='same',
               name='dec_out2'
               )(x)

    x = Dropout(rate=0.3, name='drop_out2')(x)

    x = Conv2D(filters=out_channels,
               kernel_size=1,
               activation=final_activation,
               padding='same',
               name='dec_output'
               )(x)

    return x






















