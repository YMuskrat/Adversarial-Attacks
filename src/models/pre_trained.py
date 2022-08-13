import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet50
import pathlib
import os
from src.utils.preprocessing import getImage, get_imagenet_label

class Model():
    def __init__(self,model):
        if model == 'mobileV2':
            #model initialization mobileV2
            self.model = tf.keras.applications.MobileNetV2(include_top=True,
                                                       weights='imagenet')
            self.decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
        elif model == 'resnet200':
            #Model initalization resnet200
            self.model=tf.keras.applications.resnet_rs.ResNetRS200(
                include_top=True,
                weights='imagenet',
                classes=1000,
                input_shape=None,
                input_tensor=None,
                pooling=None,
                classifier_activation='softmax',
                include_preprocessing=True)
            self.decode_predictions = tf.keras.applications.resnet_rs.decode_predictions
        self.model.trainable = False
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        
    def predict(self, image, quantisation=False,quantisation_type=None):
        """
        predict the image

        :param quantisation: checks if our network is quantised or not
        :param image: Resized processed image
        :param interpreter: loading the quantised model
        :param image_probs: image prediction
        
        :return: plot the image with the corresponding prediction
        """
  
        
        if not quantisation:
          image = getImage(image, quantisation)
          self.image_probs = self.model.predict(image)
          image_reference=image
        else:
          self.interpreter = self.load_quantised_model_tensors(quantisation_type)
          input_index = self.interpreter.get_input_details()[0]["index"]
          output_index = self.interpreter.get_output_details()[0]["index"]
          self.interpreter.set_tensor(input_index, image)
          self.interpreter.invoke()
          self.image_probs = self.interpreter.get_tensor(output_index)
          image_reference = image

        _, image_class, class_confidence = get_imagenet_label(
            self.image_probs, self.decode_predictions)
        return image_reference, image_class, class_confidence

    def load_quantised_model_tensors(self, quantisation_type):
        if quantisation_type== "16 bit":
            tf_path = r"C:\Users\yassi\OneDrive\Desktop\Adversarial_Attacks\converted_model_16.tflite"
        else:
            tf_path = r"C:\Users\yassi\OneDrive\Desktop\Adversarial_Attacks\converted_model_8.tflite"
        interpreter = tf.lite.Interpreter(
            model_path=tf_path)
        interpreter.allocate_tensors()
        return interpreter
    
    
        
