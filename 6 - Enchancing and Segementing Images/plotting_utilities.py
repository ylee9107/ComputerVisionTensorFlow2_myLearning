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

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#=========================== Plotting Functions ===========================
def plot_images_inGrid(images, titles=None, figure=None, grayscale=False, transpose=False):
    """ This builds a function to plot the Images (Noisy or otherwise) in a n x m grid.
    Parameters:
        - images, is the Images in an Array of n x m.
        - titles, is an optional List of "m" titles for each of the image columns.
        - figure, is an optional Pyplot figure (If set to None, one will be created).
        - grayscale, is an optional Flag to draw the images in grayscale.
        - transpose, is an optional Flag to transpose the grid.
    Returns:
        - returns a Pyplot figure that are filled with the images.
    """
    # Define the Rows and Columns:
    nb_cols, nb_rows = len(images), len(images[0])

    # Define the Image ratio:
    img_ratio = images[0][0].shape[1] / images[0][0].shape[0]

    # Transpose the grid, if set to True:
    if transpose:
        vert_grid_shape, hori_grid_shape = (1, nb_rows), (nb_cols, 1)
        figsize = (int(nb_rows * 5 * img_ratio), nb_cols * 5)
        wspace, hspace = 0.2, 0.
    else:
        vert_grid_shape, hori_grid_shape = (nb_rows, 1), (1, nb_cols)
        figsize = (int(nb_rows * 5 * img_ratio), nb_cols * 5)
        wspace, hspace = 0.2, 0.

    # Create Pyplot figure, if set to None:
    if figure is None:
        figure = plt.figure(figsize = figsize)

    # Flag to draw the images in grayscale:
    imshow_params = {'cmap': plt.get_cmap('gray')} if grayscale else {}

    # Grid layout to place subplots within a figure:
    grid_spec = gridspec.GridSpec(*hori_grid_shape, wspace=0, hspace=0)

    for i in range(nb_cols):
        grid_spec_i = gridspec.GridSpecFromSubplotSpec(*vert_grid_shape,
                                                       subplot_spec=grid_spec[i],
                                                       wspace=wspace,
                                                       hspace=hspace)
        for j in range(nb_rows):
            ax_img = figure.add_subplot(grid_spec_i[j])
            ax_img.set_yticks([])
            ax_img.set_xticks([])

            if titles is not None:
                if transpose:
                    ax_img.set_ylabel(titles[i], fontsize = 25)
                else:
                    ax_img.set_title(titles[i], fontsize = 15)

            ax_img.imshow(images[i][j], **imshow_params)

    figure.tight_layout()

    return figure


def figure_to_RGB_array(fig):
    """ This builds a function to convert the figure into RGB Array.
    Parameters:
        - fig, is the PyPlot Figure
    Returns:
        - returns the RGB Array.
    """
    figure_buffer = io.BytesIO()

    # Save the figure:
    fig.savefig(figure_buffer, format='png')

    figure_buffer.seek(0)

    figure_string = figure_buffer.getvalue()

    return figure_string


def figure_to_summary(fig, name, step):
    """ This builds a function to convert the Figure into TF Summary().
    Parameters:
        - fig, is the Figure.
        - name, is a Tag for the Summary Name.
    Returns:
        - returns the Summary Step.
    """
    # Transform the figure into PNG buffer:
    figure_string = figure_to_RGB_array(fig)

    # Tranform the PNG Buffer into an Image Tensor:
    figure_tensor = tf.image.decode_png(contents=figure_string,
                                        channels=4)
    figure_tensor = tf.expand_dims(input=figure_tensor,
                                   axis=0)

    # Using Proto to convert image string to Summary:
    figure_summary = tf.summary.image(name=name,
                                      data=figure_tensor,
                                      step=step)

    return figure_summary


