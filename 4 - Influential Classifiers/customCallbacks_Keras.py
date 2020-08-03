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




