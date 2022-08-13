import tensorflow_hub as hub
import tensorflow.compat.v2 as tf
import PIL.Image as Image
import numpy as np
from src.utils.preprocessing import getImage, get_imagenet_label

class Contrastive_Learning():
    def __init__(self):
        module = "https://tfhub.dev/google/supcon/resnet_v1_200/imagenet/classification/1"
        self.classifier = tf.keras.Sequential([
            hub.KerasLayer(module,input_shape=(224, 224)+(3,))])
    
    def predict(self, image, quantisation=False):
        image = getImage(image, quantisation,model="CL")
        result = self.classifier.predict(image[np.newaxis, ...])
        predicted_class = tf.math.argmax(result[0], axis=-1)
        labels_path = tf.keras.utils.get_file(
            'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        
        probabilities = tf.nn.softmax(self.classifier(
            image[np.newaxis, ...])).numpy()

        imagenet_labels = np.array(open(labels_path).read().splitlines())

        return imagenet_labels[predicted_class], probabilities[0][predicted_class.numpy()]

