import tensorflow as tf
import numpy as np
from PIL import Image


def getImage(image, quantisation, model='abc'):

        """
        Get image and process it

         :param image: process the image or normalize it and change its dimension if dealing with quantised model

        """
        if model== 'CL':
          image = Image.open(image).resize((224, 224))
          image = np.array(image)/255.0      
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


def process_image(input_image):
    """
            resize the image so the input dimensions match the output dimensions

            :param image: preprocessed resized image with dimension (1,224,224)

    """
    image_raw = tf.io.read_file(input_image)
    image = tf.image.decode_image(image_raw)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = mobilev2Processing(image)
    image = image[None, ...]
    return image
