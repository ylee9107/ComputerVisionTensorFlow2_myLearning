# Object Detection Models

## Introduction

For this project, the aim would be to go through the techniques that are used for object detection in a scene or an image. The techniques that are explored here are the __You Only Look Once (YOLO)__ and __Regions with Convolutional Neural Networks (R-CNN)__. 

The process of detecting ojects in an image or video stream coupled with their bounding boxes is what object detection. Object detection is also called object locatlisation. A bounding box is a small rectangle that surrounds the object in question/interest. Here, the input for the algorithm is usually an image and the output would be a list of bounding boces and the object classes/labels. For each of the bounding boxes, the model should be able to output the corresponding predicted class/label and its confidence that the guess it correct. 

Object detection in general are widely used in industry. For example, these models can be used in the following:
1. Self driving car - for perceiving vehicles and pedestrians.
2. Content moderation - to locate forbidden objects in the scene and its respectiv size.
3. Healthcare - detecting tumors or dangerous unwanted tissues from radiographs.
4. Manufacturing - used in assembly robots of the manufacturing chain to put together or repair products.
5. Security - to detect threats, threspasses, or count people.
6. Wildlife Conservation - to monitor the population of animals.

## Breakdown of this Notebook:
- History of the object detection techniques.
- The main approaches in object detection.
- Implementing the YOLO Architecture for fast object detection task.
- Improving upon YOLO with the Faster R-CNN architecture.
- Utilising the Faster R-CNN with the TensorFlow Object Detection API.

## Supporting Files and Utilities .py files:
- YOLOv3_WeightsConversion.py
- YOLOv3_model.py
- YOLOv3_Utilities.py
- coco.names

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

From this project, I was able to learn a great deal about the YOLO architecture and how it was able to draw bounding boxes around the objects of interests with impressive speeds.The most complex part of this project had to be the losses computation involving the bounding box coordinates and its size, confidence that the object was in the box, the scores for each of the classes and the full combination of the YOLO loss altogether. I was able to quickly implement the model inference with the webcam on a CPU only machine with the help of the OpenCV library. This project also introduced new concepts such as Average Precision Threshold, known as the Jaccard Index (or Intersection over Union, IoU) and the achor boxes was refined with a few equations and its subsequent post processing whereby using NMS to remove unwanted bounding boxes. The overall process was highly challenging and rewarding. The second part of this notebook also covered the use of the TensorFlow Object Detection API, where I was able to follow the tutorial and implement 
the Faster R-CNN model for inference either with testing images or Live Feed from the webcam.
