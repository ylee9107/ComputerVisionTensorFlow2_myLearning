"""
File name: YOLOv3_WeightsConversion.py
Author: YSLee
Date created:16.06.2020
Date last modified: 16.06.2020
Python Version: "3.7"
"""

#=========================== Import the Libraries ===========================
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS

# From custom .py files, Import the following:
from YOLOv3_model import YoloV3
from YOLOv3_Utilities import load_darknet_weights

#=========================== FLAGS ===========================
flags.DEFINE_string('weights', './yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './yolov3.tf', 'path to output')
flags.DEFINE_integer('nb_classes', 80, 'Number of classes in the model')

#=========================== Convert DarkNet model ===========================
# This will convert the DarkNet model from yolov3.weights to yolov3.tf file.

def main(_argv):

    # Instantiate the YOLO model:
    yolo_model = YoloV3(classes = FLAGS.nb_classes)
    yolo_model.summary()
    # Output logging prompt:
    logging.info('model created.')

    # Load in the Model Weights that was downloaded:
    load_darknet_weights(yolo_model, FLAGS.weights)
    # Output logging prompt:
    logging.info('weights were loaded.')

    # Do a test to check if everything is working:
    img_input = np.random.random( (1, 320, 320, 3) ).astype(np.float32)
    output = yolo_model(img_input)
    # Output logging prompt:
    logging.info("Everything checks out.")

    # Save the model weights as .tf file:
    yolo_model.save_weights(FLAGS.output)
    # Output logging prompt:
    logging.info("the model weights was successfully converted and saved as .tf file.")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass





