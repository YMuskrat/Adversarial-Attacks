import matplotlib.pyplot as plt
import tensorflow as tf
from src.utils.preprocessing import get_imagenet_label
from src.models.pre_trained import Model


class ImageHandler():
    def __init__(self):
        self.image_list=[]
        self.description_list=[]
        self.label_list=[]
        self.confidence_list = []
        self.pertubation_list = []

    def display_images(self,model_name, adv_x, image, eps, perturbations, description, decode_predictions,quantisation=False):

        """
            Helper function to display the images along with their corresponding adversarial attack pertubations

                :param get_imagenet_label: get the imagenet label of the prediction made
                :param interpreter: loading the quantised model

            return: plot showing the pertubation as well as the image
        """
        initialize_model=Model(model_name)
        pre_trained_model = initialize_model.model
        if quantisation == '32 bit':
            _, label, confidence = get_imagenet_label(pre_trained_model.predict(adv_x), decode_predictions)
        elif quantisation == '16 bit':
            _, label, confidence = initialize_model.predict(adv_x, True,quantisation)
        elif quantisation == '8 bit':
            _, label, confidence = initialize_model.predict(adv_x, True, quantisation)
        self.image_list.append(adv_x[0])
        self.description_list.append(description)
        self.label_list.append(label)
        self.confidence_list.append(confidence)
        self.pertubation_list.append(perturbations)
        return self.image_list, self.description_list, self.label_list,self.confidence_list,self.pertubation_list


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


    def image_drawer(self,image_32,description_32,label,confidence_32,pertubations_32,
                     image_16, description_16, label_16, confidence_16,
                     image_8, description_8, label_8, confidence_8): 

        zipped_results = zip(image_32, description_32, label, confidence_32, pertubations_32,
                        image_16, description_16, label_16, confidence_16,
                        image_8, description_8, label_8, confidence_8)
        for im_32, des_32, lbl_32, conf_32, pertbs_32, im_16, des_16, lbl_16, conf_16, im_8, des_8, lbl_8, conf_8 in zipped_results:
            im_32 = tf.image.flip_up_down(im_32)
            im_16 = tf.image.flip_up_down(im_16)
            im_8 = tf.image.flip_up_down(im_8)
    
            fig =plt.figure(figsize=(15,16))
            ax1 = fig.add_subplot(4,4,1)
            ax1.imshow(tf.reshape(pertbs_32, (224,224,3), name=None), aspect='auto')
            ax2 = fig.add_subplot(4,4,2)
            ax2.imshow(im_32* 0.5 + 0.5,origin='lower', extent=[-4, 4, -1, 1], aspect=4)
            plt.title('32 bit -> {} \n {} : {:.2f}% Confidence'.format(des_32,
                                                    lbl_32, conf_32*100,))
            
            ax3 = fig.add_subplot(4,4,3)
            ax3.imshow(im_16* 0.5 + 0.5,origin='lower', extent=[-4, 4, -1, 1], aspect=4)
            plt.title('16 bit ->->{} \n {} : {:.2f}% Confidence'.format(des_16,
                                                    lbl_16, conf_16*100,))
            
            ax4 = fig.add_subplot(4, 4, 4)
            ax4.imshow(im_8 * 0.5 + 0.5, origin='lower',
                       extent=[-4, 4, -1, 1], aspect=4)
            plt.title('8 bit ->->{} \n {} : {:.2f}% Confidence'.format(des_8,
                                                                 lbl_8, conf_8*100,))
            
            plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, )
            



              
