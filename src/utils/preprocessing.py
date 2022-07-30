import tensorflow as tf


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

def getImage(image, quantisation):

        """
        Get image and process it

         :param image: process the image or normalize it and change its dimension if dealing with quantised model

        """

        if not quantisation:
          image_raw = tf.io.read_file(image)
          image = tf.image.decode_image(image_raw)
          image = process_image(image)

        
        else:
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
