import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from config import image_properties

class Perceptual(object):

    def __init__(self):
        self.model = VGG19(include_top=False, weights='imagenet', input_shape=[image_properties.height,image_properties.width,image_properties.depth])

    def eval_vgg(self,x):
        for layer in self.model.layers[:5]:
            layer.trainable = False
            x=layer(x)
        return x

    def get_perceptual_loss(self,input_image, recon_image):

        input_image.set_shape([1,image_properties.height,image_properties.width,image_properties.depth])
        recon_image.set_shape([1,image_properties.height,image_properties.width,image_properties.depth])
        input_conv = self.eval_vgg(input_image)
        recon_conv = self.eval_vgg(recon_image)
        return tf.losses.mean_squared_error(input_conv,recon_conv)