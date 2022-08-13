import tensorflow as tf
from src.utils.preprocessing import process_image
from src.utils import Plotter

class Adversarial_Attack():
    def __init__(self,model):
        
        if model == 'mobileV2':
          self.model = tf.keras.applications.MobileNetV2(include_top=True,
                                                       weights='imagenet')
          self.decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
        elif model =='resnet200':
          self.model = tf.keras.applications.resnet_rs.ResNetRS200(
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
        
     
    def adversarial_attack(self, image1, adversarial_attack_type, quantization=False):
        """
            Adversarial Attack

        :param epsilons: try number of different epsilon values
        :param adv_x: generate the pertubation of the image
        :descptions: just a structered statement containing epsilon to be associated with the images at display

        :param display_images: display the image with the pertubation
        """
        #self.count = 0
        #self.image_list, self.description_list, self.label_list, self.confidence_list, self.pertubation_list = [], [], [], [], []

        if adversarial_attack_type == 'fast_FGSM':
          self.generate_adversarial_images(process_image(
              tf.image.decode_image(tf.io.read_file(image1))), adversarial_attack_type)

        #else:
          #flag = 'iterative_FGSM'
        ImageHandler=Plotter.ImageHandler()
        epsilons = [0, 0.01, 0.1, 0.15, 0.3]
        descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                        for eps in epsilons]
        for i, eps in enumerate(epsilons):
          image = tf.image.decode_image(tf.io.read_file(image1))
          # Easy Fix for now this gets our actual image
          image2 = process_image(image)
          if quantization == True:
            image = image / 225
            image = tf.image.resize(image, (224, 224))
            adv_x = self.image_pertubation_fusion(
                image, eps, adversarial_attack_type, image2)
            image_list, description_list, label_list, confidence_list, pertubation_list,count = ImageHandler.display_images(self.model,
                adv_x, descriptions[i], eps, quantization, self.perturbations, image2,self.decode_predictions,quantization[1])
          else:
            image = process_image(image)
            adv_x = self.image_pertubation_fusion(
                image, eps, adversarial_attack_type, image2)
            image_list, description_list, label_list, confidence_list, pertubation_list, count = ImageHandler.display_images(self.model,
                adv_x, descriptions[i], eps, quantization,self.perturbations , image2,self.decode_predictions,quantization[1])
          
            
         
        return image_list, description_list, label_list, confidence_list, pertubation_list, count

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
        elif adversarial_attack_type== 'iterative_FGSM':
          num_steps = 5

        signed_grad = 0
        for i in range(num_steps):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.model(input_image)
            loss = self.loss_object(input_label, prediction)
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
          signed_grad += tf.sign(gradient)
          #print(signed_grad)
        return signed_grad

    def generate_adversarial_images(self,image, adversarial_attack_type):
        """
            generating the adversarial images

            :param label_index: get an index(class number) of a largest element
            :param label: get image label
            :param pertubations: create pertubations

        """
        label_index = tf.math.argmax(self.model.predict(image), axis=1).numpy()[0]
        label = tf.one_hot(label_index, self.model.predict(image).shape[-1])
        label = tf.reshape(label, (1, self.model.predict(image).shape[-1]))
        self.perturbations = self.create_adversarial_pattern(
            image, label, adversarial_attack_type)
        
    
    def image_pertubation_fusion(self,image, eps, adversarial_attack_type, image2):
      #print('i am here',adversarial_attack_type)
      if adversarial_attack_type == 'fast_FGSM':
          # Plus pertubation and minus pertubation
          adv_x = image + eps*self.perturbations
          adv_x = tf.clip_by_value(adv_x, -1, 1)

      elif adversarial_attack_type == 'iterative_FGSM':
          label_index = tf.math.argmax(self.model.predict(image2), axis=1).numpy()[0]
          label = tf.one_hot(label_index, self.model.predict(image).shape[-1])
          label = tf.reshape(label, (1, self.model.predict(image).shape[-1]))
          #self.perturbations = self.generate_adversarial_images(
              #image2, "fast_FGSM")
          for i in range(10):
            with tf.GradientTape() as tape:
              tape.watch(image2)
              prediction = self.model(image2)
              loss = self.loss_object(label, prediction)
            # Get the gradients of the loss w.r.t to the input image.
            gradient = tape.gradient(loss, image2)
            # Get the sign of the gradients to create the perturbation
            signed_grad = 0.5 * tf.sign(gradient)
            #print(signed_grad)
            adv_temp = image2+signed_grad
            total_grad = image2 + eps * adv_temp
            adv_x = tf.clip_by_value(total_grad, -eps, eps)
            adv_x = image2 + total_grad
      return adv_x
