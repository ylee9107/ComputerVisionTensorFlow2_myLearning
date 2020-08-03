# Influential Classifiers

## Introduction

There will be 6 different Notebooks for this project and the description/introduction for each notebook are the following:

This __Notebook (1)__ will dive into more advanced developments of Convolutional Neural Networks (CNNs) that has become famous for their contributions to computer visions. The notebook will also explore methods that better prepare the CNNs to perform on specific tasks, such as the use of Transfer Learning whereby previous knowledge (trained weights) of a network on a specific use case can be transferred/re-develop for new applications.

This __Notebook (2)__ will continue on from the previous sections. This notebook will go through the process of building the __ResNet__ from scratch.

This __Notebook (3)__ will continue on from the previous sections. This notebook will go through the process of __Extension into Multiple experiments to improve model testing results__.

This __Notebook (4)__ will continue on from the previous sections. This notebook will go through the process of using __Keras Applications and reusing model__. Previously, the notebook focused on implementing the ResNet model from scratch, here, the intention is to utilise the models that were curated from Keras Applications and be used as direct comparison with the self-implemented version.

This __Notebook (5)__ will continue on from the previous sections. This notebook will go through the process of using __Getting Models from TensorFlow Hub__. Previously, the notebook focused on implementing the ResNet model from Keras Applications, here, the intention is to utilise the models that were curated from Tensorflow Hub and specifically using the Inception V3 Model.

This __Notebook (6)__ will continue on from the previous sections. This notebook will go through the process of using __Transfer Learning and Applying it__. Previously, the notebook focused on implementing the Inception model and MobieNet from TensorFlow Hub, here, the intention is to utilise transfer learning with Keras. Utilising models from Keras Applications that are pre-trained on richer datasets on new tasks. The focus here would be to fetch the parameters of pre-trained weights of the models that was trained on the ImageNet dataset, test different types of transfer learning such as freezing and fine-tuning of the feature_extractor layers.

## Breakdown of this Notebook:
- CNNs concepts.
- CNNs' relevance for Computer Vision Tasks.
- Implementation of CNNs in TensorFlow and Keras.
- Advanced optimisers and training of CNNs in an efficient manner.
- Regularisation methods to prevent overfitting.

## Supporting Utilities .py files:
In this Notebook, there will be a requirement to import the code/utilities from the following files (.py files):
-	DataPrepCIFAR_utility.py
-	customCallbacks_Keras.py


## Requirements:
- Tensorflow 2.0 or higher
- TensorFlow Datasets
- Keras (but the tf.keras version, not the standalone Keras library)
- Scikit-image
- Glob
- Numpy
- Collections
- Functools
- Math
- Matplotlib
- OS

## Summary:

From this project I was able to learn a great deal of implementation techniques and code for these models (ResNet, MobileNet, Inception etc.). In particular, I was able to learn more about the code blocks to builds the ResNet model, although the model is slightly dissimilar to the official tensorflow or Keras implementations, it still proved to be quite a performant model. Most importantly, I should be able to utilise the knowledge gained here to self-implement and experiment with future models that comes to mind or that is needed for a certain task. This project also taught me the compromises in the model architecture of MobileNet for the purpose of making it lightweight, lower latency and still being achieve a reasonable performance (compromising in some accuracy scores, therefore its generalisation). Further, I was able to experiment and discover first hand, how much better a model's performance can be due to Transfer Learning, whereby initialising a model with pretrained weights, experimenting with freezing model layers (from none to several layers) and then testing it. I believe that this project has given me invaluable skills to tackle more complex classification tasks in the future.
