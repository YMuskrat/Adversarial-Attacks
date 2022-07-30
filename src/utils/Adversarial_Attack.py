import tensorflow as tf
#from src.models.pre_trained import Model
from preprocessing import image_pertubation_fusion, process_image
from Plotter import display_images


model = Model().model  # get the model
loss_object = Model().loss_object  # get the loss object
image_probs= Model().image_probs # get image probs

def create_adversarial_pattern(self, input_image, input_label, adversarial_attack_type):
        """
        resize the image so the input dimensions match the output dimensions

        :param loss: get the loss between the label and the prediction
        :param gradient.image_probs: Get the gradients of the loss w.r.t to the input image.
        :param signed_grad: Get the sign of the gradients to create the perturbation

        return: the gradient sign for pertubation
        """
        if adversarial_attack_type == 'fast_FGSM':
          num_steps = 1
        elif adversarial_attack_type[0] == 'iterative_FGSM':
          num_steps = adversarial_attack_type[1]

        signed_grad = 0
        for i in range(num_steps):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
          signed_grad += tf.sign(gradient)
          #print(signed_grad)
        return signed_grad

def generate_adversarial_images(image,adversarial_attack_type):
        """
            generating the adversarial images

            :param label_index: get an index(class number) of a largest element
            :param label: get image label
            :param pertubations: create pertubations

        """
        label_index = tf.math.argmax(model.predict(image),axis=1).numpy()[0]
        label = tf.one_hot(label_index, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))
        perturbations = create_adversarial_pattern(image, label,adversarial_attack_type)


def adversarial_attack(self, image1, quantization, adversarial_attack_type):
        """
            Adversarial Attack

        :param epsilons: try number of different epsilon values
        :param adv_x: generate the pertubation of the image
        :descptions: just a structered statement containing epsilon to be associated with the images at display

        :param display_images: display the image with the pertubation
        """
        self.count = 0
        self.image_list, self.description_list, self.label_list, self.confidence_list, self.pertubation_list = [], [], [], [], []

        if adversarial_attack_type == 'fast_FGSM':
          perturbations=self.generate_adversarial_images(process_image(
              tf.image.decode_image(tf.io.read_file(image1))), adversarial_attack_type)

        else:
          flag = 'iterative_FGSM'

        epsilons = [0, 0.01, 0.1, 0.15, 0.3]
        descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                        for eps in epsilons]
        for i, eps in enumerate(epsilons):
          image = tf.image.decode_image(tf.io.read_file(image1))
          image2 = process_image(image)  # Easy Fix for now this gets our actual image
          if quantization == True:
            image = image / 225
            image = tf.image.resize(image, (224, 224))
            adv_x = image_pertubation_fusion(
                image, eps, adversarial_attack_type, image2, perturbations)
            image_list, description_list, label_list, confidence_list, pertubation_list = display_images(
                adv_x, descriptions[i], eps, quantization, perturbations,image2)
          else:
            image = process_image(image)
            adv_x = image_pertubation_fusion(
                image, eps, adversarial_attack_type, image2, perturbations)
            image_list, description_list, label_list, confidence_list, pertubation_list, count = display_images(
                adv_x, descriptions[i], eps, quantization,perturbations,image2)
        return image_list, description_list, label_list, confidence_list, pertubation_list, count


def image_pertubation_fusion(image, eps, adversarial_attack_type, image2, perturbations):
      if adversarial_attack_type == 'fast_FGSM':
          # Plus pertubation and minus pertubation
          adv_x = [image + eps*perturbations, image - eps*perturbations]
          adv_x = [tf.clip_by_value(adv_x[0], -1, 1),
                   tf.clip_by_value(adv_x[1], -1, 1)]

      elif adversarial_attack_type[0] == 'iterative_FGSM':
          label_index = tf.math.argmax(
              model.predict(image2), axis=1).numpy()[0]
          label = tf.one_hot(label_index, image_probs.shape[-1])
          label = tf.reshape(label, (1, image_probs.shape[-1]))
          #self.perturbations = self.create_adversarial_pattern(image, label,adversarial_attack_type)
          for i in range(adversarial_attack_type[1]):
            with tf.GradientTape() as tape:
              tape.watch(image2)
              prediction = model(image2)
              loss = loss_object(label, prediction)
            # Get the gradients of the loss w.r.t to the input image.
            gradient = tape.gradient(loss, image2)
               # Get the sign of the gradients to create the perturbation
            signed_grad = adversarial_attack_type[2] * tf.sign(gradient)
              #print(signed_grad)
            adv_temp = image2+signed_grad
            total_grad = image2 - eps * adv_temp
            adv_x = tf.clip_by_value(total_grad, -eps, eps)
            adv_x = image2 + total_grad
