# import tensorflow as tf
# import numpy as np
# import scipy.io as sio
# import os

# from config import image_properties
# class Perceptual(object):

#     def __init__(self):
#         if not os.path.isfile('imagenet-vgg-verydeep-19.mat'):
#             print('Downloading pretrained model weights for VGG19')
#             os.system('wget -b http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat')

#         self.weights = sio.loadmat('imagenet-vgg-verydeep-19.mat')['layers'][0]

#     def obtain_feat_rep(self,image):
#         layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
#             'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
#             'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4','pool3',
#             'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
#             'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5')

#         net = image
#         network = {}
#         for i,layer in enumerate(layers):
            
#             if layer.startswith('conv'):
#                 kernels,bias = self.weights[i][0][0][0][0]
#                 kernels = np.transpose(kernels,(1,0,2,3))
#                 conv = tf.nn.conv2d(net, tf.constant(kernels),strides=(1,1,1,1),padding='SAME',name=layer)
#                 net = tf.nn.bias_add(conv,bias.reshape(-1))
#                 net = tf.nn.relu(net)
#             elif layer.startswith('pool'):
#                 net = tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
            
#             network[layer] = net
#         return network

        
#     def get_perceptual_loss(self,input_image, recon_image):

#         input_image.set_shape([1,image_properties.height,image_properties.width,image_properties.depth])
#         recon_image.set_shape([1,image_properties.height,image_properties.width,image_properties.depth])
        
#         input_feat = self.obtain_feat_rep(input_image)['relu3_2']
#         recon_feat = self.obtain_feat_rep(recon_image)['relu3_2']
#         return tf.nn.l2_loss(recon_feat - input_feat) / (image_properties.height*image_properties.width*image_properties.depth)

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from config import image_properties

class Perceptual(object):

    def __init__(self):
        self.model = VGG19(include_top=False, weights='imagenet', input_shape=[image_properties.height,image_properties.width,image_properties.depth])

    def eval_vgg(self,x):
        for layer in self.model.layers[:7]:
            # layer.trainable = False
            x=layer(x)
        return x

    def get_perceptual_loss(self,input_image, recon_image):

        input_image.set_shape([1,image_properties.height,image_properties.width,image_properties.depth])
        recon_image.set_shape([1,image_properties.height,image_properties.width,image_properties.depth])
        input_conv = self.eval_vgg(input_image)
        recon_conv = self.eval_vgg(recon_image)

        return tf.reduce_mean(tf.math.squared_difference(recon_conv,input_conv) / (image_properties.height*image_properties.width*image_properties.depth))
        # return tf.losses.mean_squared_error(input_conv,recon_conv)