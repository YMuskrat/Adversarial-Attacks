import tensorflow as tf

def quantization(model, quant_type):
    """
        quantization of the model

        :param converter: convert the model to lite mode using tensorflow

        :param converter.optimizations: Convert the model to 16 bit precision
        :param self.interpreter: load the model saved

        """
    if model == 'mobileV2':
      model =tf.keras.applications.MobileNetV2(include_top=True,
                                                           weights='imagenet')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

      # Set the optimization mode
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_path = r"C:\Users\yassi\OneDrive\Desktop\Adversarial_Attacks\converted_model_8.tflite"

       # Set float16 is the supported type on the target platform
    if quant_type == "16 bit":
          converter.target_spec.supported_types = [tf.float16]
          tf_path = r"C:\Users\yassi\OneDrive\Desktop\Adversarial_Attacks\converted_model_16.tflite"

        # Convert and Save the model
    tflite_model = converter.convert()
    open(tf_path, "wb").write(tflite_model)
    
    
