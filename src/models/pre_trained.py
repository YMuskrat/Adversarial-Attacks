import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet50
import pathlib
import os
from src.utils.preprocessing import getImage

class Model():
    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2(include_top=True,
                                                       weights='imagenet')
        self.model.trainable = False
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
    
    def quantization(self, quant_type):
        """
        quantization of the model

        :param converter: convert the model to lite mode using tensorflow

        :param converter.optimizations: Convert the model to 16 bit precision
        :param self.interpreter: load the model saved

        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Set the optimization mode
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Set float16 is the supported type on the target platform
        if quant_type == "16 bit":
          converter.target_spec.supported_types = [tf.float16]

        # Convert and Save the model
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)
        self.interpreter = tf.lite.Interpreter(
            model_path="converted_model.tflite")
        self.interpreter.allocate_tensors()
    
    def predict(self, image, quantisation=False):
        """
        predict the image

        :param quantisation: checks if our network is quantised or not
        :param image: Resized processed image
        :param interpreter: loading the quantised model
        :param image_probs: image prediction
        
        :return: plot the image with the corresponding prediction
        """
  
        image = getImage(image,quantisation)
        if not quantisation:
          self.image_probs = self.model.predict(image)
          image_reference=image
        else:
          input_index = self.interpreter.get_input_details()[0]["index"]
          output_index = self.interpreter.get_output_details()[0]["index"]
          self.interpreter.set_tensor(input_index, image)
          self.interpreter.invoke()
          self.image_probs = self.interpreter.get_tensor(output_index)
          image_reference = self.image1

        _, image_class, class_confidence = self.get_imagenet_label(
            self.image_probs)
        return image_reference, image_class, class_confidence


    def get_imagenet_label(self,probs):
        """
            : return decode the predictions and take the first probable label
            """
        return self.decode_predictions(probs, top=1)[0][0]
        
