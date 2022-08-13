import matplotlib.pyplot as plt
import tensorflow as tf
from src.utils.preprocessing import get_imagenet_label
from src.models.pre_trained import Model


class ImageHandler():
    def __init__(self):
        self.count=0
        self.image_list=[]
        self.description_list=[]
        self.label_list=[]
        self.confidence_list = []
        self.pertubation_list = []

    def display_images(self,model, image, description, eps, quantization, perturbations, image2,decode_predictions,quantisation=False):

        """
            Helper function to display the images along with their corresponding adversarial attack pertubations

                :param get_imagenet_label: get the imagenet label of the prediction made
                :param interpreter: loading the quantised model

            return: plot showing the pertubation as well as the image
        """

        if quantisation == '32 bit':

            _, label, confidence = get_imagenet_label(model.predict(image), decode_predictions)
            r_image = image + eps*perturbations
            

            
        else:
            if quantisation == '16 bit':
                interpreter = tf.lite.Interpreter(
                    model_path=r"C:\Users\yassi\OneDrive\Desktop\Adversarial_Attacks\converted_model_16.tflite")
                interpreter.allocate_tensors()
            elif quantisation == '8 bit':
                interpreter = tf.lite.Interpreter(
                    model_path=r"C:\Users\yassi\OneDrive\Desktop\Adversarial_Attacks\converted_model_8.tflite")
                interpreter.allocate_tensors()

            
            input_index = interpreter.get_input_details()[0]["index"]
            output_index = interpreter.get_output_details()[0]["index"]
            interpreter.set_tensor(input_index, image)
            interpreter.invoke()
            r_image = image2 + eps*perturbations  # easy fix for now
            _, label, confidence = get_imagenet_label(
                interpreter.get_tensor(output_index), decode_predictions)

        self.image_list.append(r_image[0])
        self.description_list.append(description)
        self.label_list.append(label)
        self.confidence_list.append(confidence)
        self.pertubation_list.append(eps*perturbations[0])
        return self.image_list, self.description_list, self.label_list,self.confidence_list,self.pertubation_list,self.count


    def visualize(self,image, image_class, class_confidence):
        #image,image_class,class_confidence= self.model.predict(r_image,False)
        #image1,image_class1,class_confidence1= self.model.predict(r_image,True)
        
        
        f = plt.figure(figsize=(20,21))
        f.add_subplot(1,2, 1)
        plt.imshow(image[0] * 0.5 + 0.5)
        plt.title('32 Bit-> {} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
        #f.add_subplot(1,2, 2)
        #plt.title('{} -> {} : {:.2f}% Confidence'.format(self.quant_type,image_class1, class_confidence1*100))
        #plt.imshow(image1[0] * 0.5 + 0.5)

        # set the spacing between subplots
        #plt.subplots_adjust(left=0.1,
                        #bottom=0.1, 
                        #right=0.9, 
                        #top=0.9, 
                        #wspace=0.4, 
                        #hspace=0.4)

        plt.show(astype('uint8'))


    def image_drawer(self,image,description,label,confidence,pertubations): #image1,description1,label1,confidence1,pertubations1
        for im,des,lbl,conf,pertbs in zip(image,description,label,confidence,pertubations): #,image1,description1,label1,confidence1,pertubations1 #,im1,des1,lbl1,conf1,pertbs1
            im = tf.image.flip_up_down(im)
            #im1 = tf.image.flip_up_down(im1)

            fig =plt.figure(figsize=(15,16))
            ax1 = fig.add_subplot(4,4,1)
            ax1.imshow(pertbs,aspect='auto');
            ax2 = fig.add_subplot(4,4,2)
            ax2.imshow(im* 0.5 + 0.5,origin='lower', extent=[-4, 4, -1, 1], aspect=4)
            plt.title('32 bit -> {} \n {} : {:.2f}% Confidence'.format(des,
                                                    lbl, conf*100,))
            """
            ax3 = fig.add_subplot(4,4,3)
            ax3.imshow(pertbs1,aspect='auto');
            ax4 = fig.add_subplot(4,4,4)
            ax4.imshow(im1* 0.5 + 0.5,origin='lower', extent=[-4, 4, -1, 1], aspect=4)
            plt.title('{}->{} \n {} : {:.2f}% Confidence'.format(self.quant_type,des1,
                                                    lbl1, conf1*100,))
            
            plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
            """



              
