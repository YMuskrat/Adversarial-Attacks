from preprocessing import get_imagenet_label

from src.models.pre_trained import Model


count = 0
image_list =[]

description_list = []

label_list = []

confidence_list = []

pertubation_list = []


def display_images(image, description, eps, quantization, perturbations, image2):
    model = Model().model  # get the model

    interpreter = Model().interpreter
    """
        Helper function to display the images along with their corresponding adversarial attack pertubations

            :param get_imagenet_label: get the imagenet label of the prediction made
            :param interpreter: loading the quantised model

        return: plot showing the pertubation as well as the image
    """

    if not quantization:
            _, label, confidence = get_imagenet_label(model.predict(image[0]))
            __, label1, confidence1 = get_imagenet_label(
                model.predict(image[0]))
            if label == label and confidence1 >= confidence:
              count += 1
            image = image2 + eps*perturbations

    else:
          input_index = interpreter.get_input_details()[0]["index"]
          output_index = interpreter.get_output_details()[0]["index"]
          interpreter.set_tensor(input_index, image)
          interpreter.invoke()
          image = image2 + eps*perturbations  # easy fix for now
          _, label, confidence = get_imagenet_label(
              interpreter.get_tensor(output_index))

    image_list.append(image[0])
    description_list.append(description)
    label_list.append(label)
    confidence_list.append(confidence)
    pertubation_list.append(eps*perturbations[0])
    return image_list, description_list, label_list,confidence_list,pertubation_list,count
