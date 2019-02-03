import tensorflow as tf
import os
from scipy.io import loadmat

class Perceptual(object):

    def __init__(self):
############################################ file name for weights ##############################
        if not os.path.isfile('./imagenet-vgg-verydeep-19.mat'):
            print('Downloading pretrained model for VGG19 for perceptual loss')
            os.system('wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat')

        self.weights_VGG = loadmat('./imagenet-vgg-verydeep-19.mat')
        print(self.weights_VGG)


        




    # @staticmethod
    # def perceptual_loss(input_image, recon_image):
    #     pass
        


if __name__ == '__main__':
    Perceptual()