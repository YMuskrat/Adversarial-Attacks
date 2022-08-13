import tensorflow as tf
import numpy as np
from PIL import Image



def process_image(input_image):
        """
            resize the image so the input dimensions match the output dimensions

            :param image: preprocessed resized image with dimension (1,224,224)

            """
        image = tf.cast(input_image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = mobilev2Processing(image)
        image = image[None, ...]
        return image


def getImage(image, quantisation, model='abc'):

        """
        Get image and process it

         :param image: process the image or normalize it and change its dimension if dealing with quantised model

        """
        if model== 'CL':
          image = Image.open(image).resize((224, 224))
          image = np.array(image)/255.0
        else:

            if not quantisation: # in case of quantisation
                image_raw = tf.io.read_file(image)
                image = tf.image.decode_image(image_raw)
                image = process_image(image)

            else: # when the network is quantised
                image =tf.image.decode_image(tf.io.read_file(image))
                image1 = process_image(image) #easy fix for now
                image = image / 225
                image=tf.image.resize(image, (224,224))
                image = tf.expand_dims(image, axis=0) 
      
        return image


def mobilev2Processing(image):
    """
        :return The input pixel values are sample-wise scaled from -1 to 1.

    """
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def get_imagenet_label(probs,decode_predictions):
        """
            : return decode the predictions and take the first probable label
        """
        return decode_predictions(probs, top=1)[0][0]
