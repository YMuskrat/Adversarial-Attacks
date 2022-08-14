import tensorflow as tf
import numpy as np
from PIL import Image


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


def process_image(input_image, model="Fully_Supervised_Learning"):
    """
             resize the image so the input dimensions match the output dimensions
            :param image: preprocessed resized image with dimension (1,224,224)
    """
    if model == "Contrastive_Supervised_Learning":
        image = Image.open(input_image).resize((224, 224))
        image = np.array(image)/255.0
        image=image[np.newaxis, ...]
    else:
        image_raw = tf.io.read_file(input_image)
        image = tf.image.decode_image(image_raw)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = mobilev2Processing(image)
        image = image[None, ...]
    return image
