"""
File name: customCallbacks_Keras.py
Author: YSLee
Date created: 01.06.2020
Date last modified: 01.06.2020
Python Version: "3.7"
"""
#=========================== Import the Libraries ===========================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from plotting_utilities import plot_images_inGrid, figure_to_summary

#=========================== Defined Log Variables ===========================

# Setting some variables to format the logs:
log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\033[94m', '\033[92m'
log_begin_bold, log_begin_underline = '\033[1m', '\033[4m'
log_end_format = '\033[0m'

#=========================== Custon Keras Callbacks Functions ===========================

class Simplified_LogCallback(tf.keras.callbacks.Callback):
    """ This builds the Keras Callbacks for a more simpler and concise console logs."""

    def __init__(self, metrics_dict, nb_epochs='?', log_frequency=1,
                 metric_string_template='\033[1m[[name]]\033[0m = \033[94m{[[value]]:5.3f}\033[0m'):
        """ This is the Initialisation of the Callback.
        Parameters:
            - metrics_dict, is the Dict containing the mappings for metrics names(or keys),
                    e.g. {"accuracy": "acc", "val. accuracy": "val_acc"}.
            - nb_epochs, is the number of training epochs.
            - log_frequency, is the frequency that the logs will be printed in epochs.
            - metric_string_template, is an optional Sttring template to print each of the metric.
        """
        # Initialise and Inherit "tf.keras.callbacks.Callback":
        super().__init__()

        self.metrics_dict = collections.OrderedDict(metrics_dict)
        self.nb_epochs = nb_epochs
        self.log_frequency = log_frequency

        # Build the format for printing out the metrics:
        # e.g. "Epoch 0/9: loss = 1.00; val-loss = 2.00"
        log_string_template = 'Epoch {0:2}/{1}: '
        separator = '; '

        i = 2
        for metric_name in self.metrics_dict:
            templ = metric_string_template.replace('[[name]]', metric_name).replace('[[value]]', str(i))
            log_string_template += templ + separator
            i += 1

        # Remove the "; " (separator) after the last element:
        log_string_template = log_string_template[:-len(separator)]
        self.log_string_template = log_string_template

    def on_train_begin(self, logs=None):
        print("Training: {}start{}".format(log_begin_red, log_end_format))

    def on_train_end(self, logs=None):
        print("Training: {}end{}".format(log_begin_green, log_end_format))

    def on_epoch_end(self, epoch, logs={}):
        if (epoch - 1) % self.log_frequency == 0 or epoch == self.nb_epochs:
            values = [logs[self.metrics_dict[metric_name]] for metric_name in self.metrics_dict]
            print(self.log_string_template.format(epoch, self.nb_epochs, *values))


class TensorBoard_ImageGrid_Callback(tf.keras.callbacks.Callback):
    """ This builds the Keras callback class for Generative models,
        to draw grids of input/predicted/target iamges into TensorBoard for every epoch.
        It also inherits the properties of Keras Callbacks.
    """

    def __init__(self, log_dir, input_images, target_images=None, tag='images',
                 figsize=(10, 10), dpi=300, grayscale=False, transpose=False, preprocess_fn=None):
        """ Initialises the Callback.
        Parameters:
            - log_dir, is the pathway to Folder to write the image summaries in.
            - input_images, is the List of input images for the grid.
            - target_images, is an optional List of target images for the grid.
            - tag, is the Tag to name the TensorBoard summary.
            - figsize, specifies the Pyplot figure size for the grid.
            - dpi, is the Pyplot figure DPI, higher the value the better the resolution.
            - grayscale, is a Flag to plot in grayscale.
            - transpose, is a Flag to transpose the grid of images.
            - preprocess_fn, is an optional function to preprocess the input/predicted/target Lists of Images before plotting.
        """
        super().__init__()

        self.summary_writer = tf.summary.create_file_writer(log_dir)

        self.input_images, self.target_images = input_images, target_images
        self.tag = tag
        self.postprocess_fn = preprocess_fn

        self.image_titles = ['images', 'predicted']
        if self.target_images is not None:
            self.image_titles.append('ground-truth')

        # Initialise the figure:
        self.fig = plt.figure(num=0,
                              figsize=figsize,
                              dpi=dpi)
        self.grayscale = grayscale
        self.transpose = transpose

    def on_epoch_end(self, epoch, logs={}):
        """ This finction will plot into TensorBoard a grid of image results.
        Parameters:
            - epoch, is the epoch number.
            - logs, is a/an (unused) Dictionary of loss/metrics value for the epoch.
        """
        # Grab the predictions with the Current Model:
        predicted_images = self.model.predict_on_batch(self.input_images)

        if self.postprocess_fn is not None:
            input_images, predicted_images, target_images = self.postprocess_fn(self.input_images,
                                                                                predicted_images,
                                                                                self.target_images)
        else:
            input_images, target_images = self.input_images, self.target_images

        # Fill the Figure with the Images:
        grid_imgs = [input_images, predicted_images]

        if target_images is not None:
            grid_imgs.append(target_images)

        self.fig.clf()
        self.fig = plot_images_inGrid(images=grid_imgs,
                                      titles=self.image_titles,
                                      figure=self.fig,
                                      grayscale=self.grayscale,
                                      transpose=self.transpose)

        with self.summary_writer.as_default():
            # Transform the Pyplot Figure into TensorFlow Summary()
            figure_summary = figure_to_summary(fig=self.fig,
                                               name=self.tag,
                                               step=epoch)

        # Log it: Forces summary writer to send any buffered data to storage.
        self.summary_writer.flush()

    def on_train_end(self, logs={}):
        """ This is a function to close the resources being used to plot the grids.
        Parameters:
            - logs, is a/an (unused) Dictionary of loss/metrics value for the epoch.
        """
        self.summary_writer.close()
        plt.close(self.fig)



