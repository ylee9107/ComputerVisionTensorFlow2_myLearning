# TensorFlow Basics and Model Training

## Introduction

TensorFlow (TF) is a numerical processing library that is widely used by machine learning and deep learning researchers or practitioners. TF is used for training and running deep neural networks. This project aims to cover the introduction of TF version 1.x and 2.0.

## Breakdown of this Notebook:
- Introduction to TensorFlow 2 and Keras packages
- Creating and Training a simple Computer Vision (CV) model.
- Tensorflow and Keras core concepts (Building Keras Models and Layers in Different Ways)
- The TensorFlow ecosystem.

## Dataset:

The dataset can be obtain from the link: http://yann.lecun.com/exdb/mnist/

The MNIST Digits dataset contains 70,000 greyscale images that have 28 x 28 pixels for each of the image. This dataset has been a reference set over the last few years to test and improve methods for this recognition task. The Input vector for the network works out to be 28 x 28 = 784 values and it has an output of 10 values (where there are 10 different digits ranging from 0 to 9). Further, the number of hidden layers for this network will be up to the modeller. 

## Requirements:
1. TensorFlow 2.0
2. If running on GPU, requires the CUDA enabled "tensorflow-gpu"

## Summary:

From this project, I was able to learn a lot about Tensorflow 1 and 2 as well as its inner workings. I able to cover concepts of TF2 on tensors, graphs, AutoGraphs, its execution type (Eager and Lazy) and gradient tape. Additionally, advanced concepts such as "tf.functions", Variables, Estimator APIs and Distribution strategies were also covered. Learning about these have broaden my knowledge about TF2. Other than these concepts, I was also able to build a basic CV model with the Keras API (and also applied it with the Estimator API) to classify MNIST Digits images. I was also able to learn about the main tools for DL development with TF such as TensorBoard for monitoring purposes, TFX for preprocessing and analysis of the model. 
