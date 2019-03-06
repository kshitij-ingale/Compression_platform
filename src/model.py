#!/usr/bin/python3
# Script to build computational graph for GAN based compression
# Code borrowed from Justin Tan (https://github.com/Justin-Tan/generative-compression) and modified as required

import tensorflow as tf
import time, os

from config import directories
from data import Data
from network import Network
from perceptual import Perceptual

class Model():
    def __init__(self, config, paths,  name='gan_compression', evaluate=False):
        """
        Function to build computational graph

        Input:
        config   : config class static variables for using user defined configuration 
        paths    : Dataframe consisting of location of training dataset images
        name     : Name for the model 
        evaluate : Variable to check if model is used for inference (True) or training (False)

        Output:
        None (Function will be used to initiate model instance)
        """

        print('Building computational graph')
        self.G_global_step = tf.Variable(0, trainable=False)
        self.D_global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)
        
        self.path_placeholder = tf.placeholder(paths.dtype, paths.shape)
        self.test_path_placeholder = tf.placeholder(paths.dtype, paths.shape)
        
        # Loading Dataset
        train_dataset = Data.load_dataset(self.path_placeholder, config.batch_size)
        test_dataset = Data.load_dataset(self.test_path_placeholder, config.batch_size, test=True)

        # Initiating iterators for traversing through dataset
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, train_dataset.output_types, train_dataset.output_shapes)
        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()
        
        # =======================================================================================================================================
        # Encoding and decoding section
        
        if evaluate:
            self.example = self.test_iterator.get_next()
        else:
            self.example = self.train_iterator.get_next()

        # with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('generator'):
            # Encode input image to obtain latent features map
            self.feature_map = Network.encoder(self.example, config, self.training_phase, config.channel_bottleneck)
            # Quantize latent feature map for lossy compression
            self.z = Network.quantizer(self.feature_map, config)
            # Reconstruct compressed image from quantized vector
            self.reconstruction = Network.decoder(self.z, config, self.training_phase, C=config.channel_bottleneck)

        print('Real image shape:', self.example.get_shape().as_list())
        print('Reconstruction shape:', self.reconstruction.get_shape().as_list())

        # Inference model developed till this point in graph
        if evaluate:
            return

        # =======================================================================================================================================
        # Adding discriminator to graph
        if config.multiscale:
            D_x, D_x2, D_x4, *Dk_x = Network.multiscale_discriminator(self.example, config, self.training_phase, 
                use_sigmoid=config.use_vanilla_GAN, mode='real')
            D_Gz, D_Gz2, D_Gz4, *Dk_Gz = Network.multiscale_discriminator(self.reconstruction, config, self.training_phase, 
                use_sigmoid=config.use_vanilla_GAN, mode='reconstructed', reuse=True)
        else:
            D_x = Network.discriminator(self.example, config, self.training_phase, use_sigmoid=config.use_vanilla_GAN)
            D_Gz = Network.discriminator(self.reconstruction, config, self.training_phase, use_sigmoid=config.use_vanilla_GAN, reuse=True)
         
        # =======================================================================================================================================
        # Loss functions
        if config.use_vanilla_GAN is True:
            D_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_x,
                labels=tf.ones_like(D_x)))
            D_loss_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_Gz,
                labels=tf.zeros_like(D_Gz)))
            self.D_loss = D_loss_real + D_loss_gen
            self.G_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_Gz,
                labels=tf.ones_like(D_Gz)))
        else:
            self.D_loss = tf.reduce_mean(tf.square(D_x - 1.)) + tf.reduce_mean(tf.square(D_Gz))
            self.G_loss = tf.reduce_mean(tf.square(D_Gz - 1.))

            if config.multiscale:
                self.D_loss += tf.reduce_mean(tf.square(D_x2 - 1.)) + tf.reduce_mean(tf.square(D_x4 - 1.))
                self.D_loss += tf.reduce_mean(tf.square(D_Gz2)) + tf.reduce_mean(tf.square(D_Gz4))

        # Pixel based loss function
        distortion_penalty = config.distortion_coeff * tf.losses.mean_squared_error(self.example, self.reconstruction)
        self.G_loss += distortion_penalty
        
        # Perceptual loss based on VGG network features
        per_loss = Perceptual(self.example.shape)
        perceptual_loss = config.perceptual_coeff * per_loss.get_perceptual_loss(self.example, self.reconstruction)
        self.G_loss += perceptual_loss
        
        # Feature matching loss using downsampled images
        if config.use_feature_matching_loss:
            D_x_layers, D_Gz_layers = [j for i in Dk_x for j in i], [j for i in Dk_Gz for j in i]
            feature_matching_loss = tf.reduce_sum([tf.reduce_mean(tf.abs(Dkx-Dkz)) for Dkx, Dkz in zip(D_x_layers, D_Gz_layers)])
            self.G_loss += config.feature_matching_weight * feature_matching_loss

        # =======================================================================================================================================
        # Setting optimizer for the loss functions
        G_opt = tf.train.AdamOptimizer(learning_rate=config.G_learning_rate, beta1=0.5)
        D_opt = tf.train.AdamOptimizer(learning_rate=config.D_learning_rate, beta1=0.5)

        # Compiling loss function to minimize loss
        def scope_variables(name):
            # with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name):
                return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        theta_G = scope_variables('generator')
        theta_D = scope_variables('discriminator')
        G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        with tf.control_dependencies(G_update_ops):
            self.G_opt_op = G_opt.minimize(self.G_loss, name='G_opt', global_step=self.G_global_step, var_list=theta_G)
        with tf.control_dependencies(D_update_ops):
            self.D_opt_op = D_opt.minimize(self.D_loss, name='D_opt', global_step=self.D_global_step, var_list=theta_D)

        G_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.G_global_step)
        G_maintain_averages_op = G_ema.apply(theta_G)
        D_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.D_global_step)
        D_maintain_averages_op = D_ema.apply(theta_D)

        with tf.control_dependencies(G_update_ops+[self.G_opt_op]):
            self.G_train_op = tf.group(G_maintain_averages_op)
        with tf.control_dependencies(D_update_ops+[self.D_opt_op]):
            self.D_train_op = tf.group(D_maintain_averages_op)

        # =======================================================================================================================================
        # Tensorboard summary parameters
        # Loss terms for tensorboard summary
        tf.summary.scalar('generator_loss', self.G_loss)
        tf.summary.scalar('discriminator_loss', self.D_loss)
        tf.summary.scalar('distortion_penalty', distortion_penalty)
        tf.summary.scalar('perceptual_loss', perceptual_loss)
        if config.use_feature_matching_loss:
            tf.summary.scalar('feature_matching_loss', feature_matching_loss)

        # Image quality metrics PSNR and SSIM for evaluating performance
        psnr = tf.image.psnr(self.example,self.reconstruction,max_val=1.0)[0]
        tf.summary.scalar('PSNR', psnr)
        ssim = tf.image.ssim(self.example,self.reconstruction,max_val=1.0)[0]
        tf.summary.scalar('SSIM', ssim)
        
        # Input and reconstructed images for tensorboard summary
        tf.summary.image('real_images', self.example[:,:,:,:3], max_outputs=4)
        tf.summary.image('compressed_images', self.reconstruction[:,:,:,:3], max_outputs=4)
        
        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_train_{}'.format(name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_test_{}'.format(name, time.strftime('%d-%m_%I:%M'))))
