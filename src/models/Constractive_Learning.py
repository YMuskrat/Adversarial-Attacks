import tensorflow_hub as hub
import tensorflow.compat.v2 as tf
import PIL.Image as Image
import numpy as np
from src.utils.preprocessing import process_image

class Contrastive_Learning():
    def __init__(self):
        module = "https://tfhub.dev/google/supcon/resnet_v1_200/imagenet/classification/1"
        self.classifier = tf.keras.Sequential([hub.KerasLayer(module,input_shape=(224, 224)+(3,))])
        self.image_list = []
        self.label_list = []
        self.confidence_list = []
    def predict(self, image):
        image=process_image(image, 'Contrastive_Supervised_Learning')
        labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        probabilities = tf.nn.softmax(self.classifier(image)).numpy()
        imagenet_labels = np.array(open(labels_path).read().splitlines())
        top_5 = tf.argsort(probabilities, axis=-1, direction="DESCENDING")[0][:5].numpy()
        self.image_list.append(tf.convert_to_tensor(image))
        self.label_list.append(imagenet_labels[top_5[0]])
        self.confidence_list.append(probabilities[0][top_5[0]])
        return self.image_list, self.label_list, self.confidence_list

