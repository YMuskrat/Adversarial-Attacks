import tensorflow as tf
from src.utils.preprocessing import process_image
from src.utils import Plotter
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from src.models.pre_trained import Model
import numpy as np

class Adversarial_Attack():
    def __init__(self,model_name):
        self.model_name, model_ini = model_name, Model(model_name)
        self.model = model_ini.model
        self.decode_predictions=model_ini.decode_predictions   
    def adversarial_attack(self, image1, adversarial_attack_type, quantization=False):
        """
            Adversarial Attack

        :param epsilons: try number of different epsilon values
        :param adv_x: generate the pertubation of the image
        :descptions: just a structered statement containing epsilon to be associated with the images at display

        :param display_images: display the image with the pertubation
        """
        ImageHandler = Plotter.ImageHandler()
        epsilons = [0.01,0.1, 0.15, 0.3]
        descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                        for eps in epsilons]
        for i, eps in enumerate(epsilons):
          if adversarial_attack_type== 'FSGM':
            image=process_image(image1)
            adv_x = fast_gradient_method(self.model, image, eps, np.inf)
            image_list, description_list, label_list, confidence_list, pertubation_list=ImageHandler.display_images(
                self.model_name, adv_x, image, eps, image-adv_x,descriptions[i], self.decode_predictions, quantization[1])
          else:
            image = process_image(image1)
            adv_x = projected_gradient_descent(self.model,image,eps=eps,eps_iter=0.0005,nb_iter=10,norm=np.inf)
            image_list, description_list, label_list, confidence_list, pertubation_list=ImageHandler.display_images(
                self.model_name,adv_x, image,eps, image-adv_x ,descriptions[i], self.decode_predictions, quantization[1])
        return image_list, description_list, label_list, confidence_list, pertubation_list


