#!/usr/bin/python3
# Script for perceptual loss using VGG19 network

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19

class Perceptual(object):

    def __init__(self,input_shape):
        """
        Function to build pretrained VGG19 network

        Input:
        input_shape : Pretrained VGG19 input image shape

        Output:
        None (Function will be used to initiate model instance)
        """

        self.model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape[1:])

    def eval_vgg(self,x):
        """
        Function to obtain pretrained VGG19 features for input image

        Input:
        x : Input image

        Output:
        x : Features vector for input image at output of 7th layer of the network
        """

        for layer in self.model.layers[:7]:
            layer.trainable = False
            x=layer(x)
        return x

    def get_perceptual_loss(self,input_image, recon_image):
        """
        Function to evaluate perceptual loss

        Input:
        input_image : Input image for which feature vector is to be generated
        recon_image : reconstructed image for which feature vector is to be generated

        Output:
        Perceptual loss evaluated as RMSE between feature vector of input image and reconstructed image
        """

        input_conv = self.eval_vgg(input_image)
        recon_conv = self.eval_vgg(recon_image)
            
        return tf.sqrt(tf.losses.mean_squared_error(input_conv,recon_conv))
