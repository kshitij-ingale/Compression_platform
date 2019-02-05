import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from config import image_properties

class Perceptual(object):

    def __init__(self,input_shape):
        self.model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape[1:])
        print(self.model.summary())

    def eval_vgg(self,x):
        for layer in self.model.layers[:7]:
            layer.trainable = False
            x=layer(x)
        return x

    def get_perceptual_loss(self,input_image, recon_image):
        
        input_conv = self.eval_vgg(input_image)
        recon_conv = self.eval_vgg(recon_image)
            
        return tf.losses.mean_squared_error(input_conv,recon_conv)
