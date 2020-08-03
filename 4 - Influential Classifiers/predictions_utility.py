"""
File name: predictions_utility.py
Author: YSLee
Date created: 01.06.2020
Date last modified: 01.06.2020
Python Version: "3.7"

"""

#=========================== Import the Libraries ===========================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#=========================== Custon Keras Callbacks Functions ===========================

def load_image(image_path, size):
    """ This builds an Image Loader where it converts the images into Numpy Arrays.
    Parameters:
        - image_path, is the directory pathway to the images.
        - size, is the target size of the images.
    Returns:
        - returns an Image Array, that is normalised between 0 and 1.
    """
    image = img_to_array(load_img(image_path, target_size=size)) / 255.
    return image


def process_predictions(class_probabilities, class_readable_labels, k=5):
    """ This builds a Batch of predictions from the Estimator.
    Parameters:
        - class_probabilities, is the prediction results that is returned by the Keras Classifier for a batch of data.
        - class_readable_labels, is the List of readable class labels that is used for display.
        - k, is the number of top predictions in/for consideration.
    Returns:
        - returns the readable labels and probabilities for the predicted classes.
    """
    # Define an empty array for the Labels and Probabilities:
    topk_labels = []
    topk_probabilities = []
    top5_probabilities = []

    # Loop through the (sorted for top 5) predictions:
    for i in range(len(class_probabilities)):
        # Grab the top-k predictions: "-k" takes the first 5 results.
        topk_classes = sorted(np.argpartition(class_probabilities[i], -k)[-k:])

        # Get the corresponding labels and predictions:
        topk_labels.append([class_readable_labels[predicted] for predicted in topk_classes])

        # Update the List while,
        # Convert Eager Tensor to Numpy Array: for compatibility of loop iterations.
        class_probabilities = np.array(class_probabilities)
        topk_probabilities.append(class_probabilities[i][topk_classes])

    # Return the labels and its probabilities:
    return topk_labels, topk_probabilities


def display_predicitons(images, topk_labels, topk_probabilities):
    """ This build a Plotting function to display the batch of predictions.
    Parameters:
        - iamges, is the Batch of input images.
        - topk_labels, is the String labels of the predicted classes.
        - topk_probabilities, is the probabilities for each of the classes.
    """
    # Define the Plot size with number of images:
    nb_images = len(images)
    nb_images_sqrt = np.sqrt(nb_images)
    plot_cols = plot_rows = int(np.ceil(nb_images_sqrt))

    # Plotting:
    figure = plt.figure(figsize=(13, 10))
    grid_spec = gridspec.GridSpec(plot_cols, plot_rows)

    for i in range(nb_images):
        img, pred_labels, pred_probs = images[i], topk_labels[i], topk_probabilities[i]

        # Resize the labels for better fit with plots:
        pred_labels = [label.split(',')[0][:20] for label in pred_labels]

        # Customise layout for multiple Axes in a grid-like pattern within a figure:
        grid_spec_idx = gridspec.GridSpecFromSubplotSpec(nrows=3,
                                                         ncols=1,
                                                         subplot_spec=grid_spec[i],
                                                         hspace=0.1)

        # Plot the images itself:
        ax_img = figure.add_subplot(grid_spec_idx[:2])
        ax_img.axis('off')
        ax_img.imshow(img)
        ax_img.autoscale(tight=True)

        # Plot the bar chart for each of the predicitons:
        ax_pred = figure.add_subplot(grid_spec_idx[2])
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.spines['bottom'].set_visible(False)
        ax_pred.spines['left'].set_visible(False)

        y_pos = np.arange(len(pred_labels))

        ax_pred.barh(y_pos, pred_probs, align='center')
        ax_pred.set_yticks(y_pos)
        ax_pred.set_yticklabels(pred_labels)
        ax_pred.invert_yaxis()

    plt.tight_layout()
    plt.show()

