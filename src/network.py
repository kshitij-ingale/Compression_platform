#!/usr/bin/python3
# Script to build network structure for encoder, quantizer, decoder and discriminator modules of model
# Code borrowed from Justin Tan (https://github.com/Justin-Tan/generative-compression) and modified as required

import tensorflow as tf
from utils import Utils
from config import input_attributes

#============================================================================================================================
# Network class for architectures, convolutional, upsampling blocks used in the class at the end
class Network(object):
    
    @staticmethod
    def encoder(x, config, training, C, reuse=False, actv=tf.nn.relu, scope='image'):
        """
        Function to build encoder architecture for encoding image (H,W,d) to latent feature map (H/16,W/16,C)
        
        Input:
        x        : Input image
        config   : Configuration parameters
        training : Variable to check for training phase
        C        : Compression bottleneck (Number of latent factors to be considered)
        reuse    : Reuse variable scope flag
        actv     : Activation function for convolutional layers
        scope    : Variable scope for the encoder block
        
        Output:
        feature_map : Latent features of the encoded image
        """
        
        init = tf.contrib.layers.xavier_initializer()
        print('Building global {} generator architecture'.format(scope))

        with tf.variable_scope('encoder_{}'.format(scope), reuse=reuse):

            # Encode image to downsampled dimensions (H,W) -> (H/16, W/16)
            f = [60, 120, 240, 480, 960]
            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            out = conv_block(x, filters=f[0], kernel_size=7, strides=1, padding='VALID', actv=actv)
            print("Encoder downsampled vector shape - ", out.shape)
            out = conv_block(out, filters=f[1], kernel_size=3, strides=2, actv=actv)
            print("Encoder downsampled vector shape - ", out.shape)
            out = conv_block(out, filters=f[2], kernel_size=3, strides=2, actv=actv)
            print("Encoder downsampled vector shape - ", out.shape)
            out = conv_block(out, filters=f[3], kernel_size=3, strides=2, actv=actv)
            print("Encoder downsampled vector shape - ", out.shape)
            out = conv_block(out, filters=f[4], kernel_size=3, strides=2, actv=actv)
            print("Encoder downsampled vector shape - ", out.shape)

            # Obtain compressed image latent vectors as per compression bottleneck
            out = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            feature_map = conv_block(out, filters=C, kernel_size=3, strides=1, padding='VALID', actv=actv)
            print("Encoded feature map shape        - ", feature_map.shape)
            return feature_map


    @staticmethod
    def quantizer(w, config, reuse=False, temperature=1, scope='image'):
        
        """
        Function to build quantizer architecture for quantizing feature vector to discrete levels 
        
        Input:
        w           : Latent features of the encoded image
        config      : Configuration parameters
        reuse       : Reuse variable scope flag
        temperature : Damping factor of quantization
        scope       : Variable scope for the quantizer block
        
        Output:
        w_bar : Quantized representation of latent features
        """
        with tf.variable_scope('quantizer_{}'.format(scope, reuse=reuse)):
            # Quantization centers
            centers = tf.cast(tf.range(-(config.quant_centers//2),(config.quant_centers//2)+1), tf.float32)
            
            # Quantize w to discrete levels as defined in centers
            w_stack = tf.stack([w for _ in range(config.quant_centers)], axis=-1)
            w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)

            # Soft quantization part for w
            smx = tf.nn.softmax(-1.0/temperature * tf.abs(w_stack - centers), dim=-1)
            w_soft = tf.einsum('ijklm,m->ijkl', smx, centers)
            
            # Representing quantization as differentiable function for optimization
            w_bar = tf.round(tf.stop_gradient(w_hard - w_soft) + w_soft)

            return w_bar


    @staticmethod
    def decoder(w_bar, config, training, C, reuse=False, actv=tf.nn.relu, channel_upsample=960):
        """
        Function to build quantizer architecture for quantizing feature vector to discrete levels 
        
        Input:
        w_bar            : Quantized representation of latent features
        config           : Configuration parameters
        training         : Variable to check for training phase
        C                : Compression bottleneck (Number of latent factors to be considered)
        reuse            : Reuse variable scope flag
        actv             : Activation function for convolutional layers
        channel_upsample : Upsample compressed vector factor
        
        Output:
        out : Reconstructed image from compressed vector
        """
        
        init = tf.contrib.layers.xavier_initializer()

        # Upsample compressed latent vector
        with tf.variable_scope('decoder', reuse=reuse):
            print("Decoder input layer shape      - ", w_bar.shape)
            w_bar = tf.pad(w_bar, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            upsampled = conv_block(w_bar, filters=960, kernel_size=3, strides=1, padding='VALID', actv=actv)

            # Upsample compressed latent vector using residual blocks
            res = residual_block(upsampled, 960, actv=actv, training=training)
            res = residual_block(res, 960, actv=actv, training=training)
            res = residual_block(res, 960, actv=actv, training=training)
            res = residual_block(res, 960, actv=actv, training=training)
            res = residual_block(res, 960, actv=actv, training=training)
            res = residual_block(res, 960, actv=actv, training=training)
            res = residual_block(res, 960, actv=actv, training=training)
            res = residual_block(res, 960, actv=actv, training=training)
            res = residual_block(res, 960, actv=actv, training=training)

            # Upsample to input image dimensions
            f = [480, 240, 120, 60]
            print("Decoder upsampled vector shape - ", res.shape)
            ups = upsample_block(res, f[0], 3, strides=[2,2], padding='same', training=training)
            print("Decoder upsampled vector shape - ", ups.shape)
            ups = upsample_block(ups, f[1], 3, strides=[2,2], padding='same', training=training)
            print("Decoder upsampled vector shape - ", ups.shape)
            ups = upsample_block(ups, f[2], 3, strides=[2,2], padding='same', training=training)
            print("Decoder upsampled vector shape - ", ups.shape)
            ups = upsample_block(ups, f[3], 3, strides=[2,2], padding='same', training=training)
            print("Decoder upsampled vector shape - ", ups.shape)

            ups = tf.pad(ups, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            ups = tf.layers.conv2d(ups, input_attributes.DEPTH, kernel_size=7, strides=1, padding='VALID')
            print("Decoder upsampled vector shape - ", ups.shape)
            out = tf.nn.tanh(ups)

            return out


    @staticmethod
    def discriminator(x, config, training, reuse=False, actv=tf.nn.leaky_relu, use_sigmoid=False, ksize=4):
        """
        Function to build discriminator architecture for differentiating between real and reconstructed images
        
        Input:
        x                : Input image
        config           : Configuration parameters
        training         : Variable to check for training phase
        reuse            : Reuse variable scope flag
        actv             : Activation function for convolutional layers
        use_sigmoid      : Variable to check if vanilla GAN structure is to be used
        ksize            : Kernel size for convolutional layers
        
        Output:
        out : Classification between real and reconstructed image
        """
        
        in_kwargs = {'center':True, 'scale':True, 'activation_fn':actv}

        print('Shape of input image x :', x.get_shape().as_list())

        with tf.variable_scope('discriminator', reuse=reuse):

            c1 = tf.layers.conv2d(x, 64, kernel_size=ksize, strides=2, padding='same', activation=actv)
            c2 = conv_block(c1, filters=128, kernel_size=ksize, strides=2, padding='same', actv=actv)
            c3 = conv_block(c2, filters=256, kernel_size=ksize, strides=2, padding='same', actv=actv)
            c4 = conv_block(c3, filters=512, kernel_size=ksize, strides=2, padding='same', actv=actv)
            out = tf.layers.conv2d(c4, 1, kernel_size=ksize, strides=1, padding='same')

            if use_sigmoid is True:
                out = tf.nn.sigmoid(out)

        return out


    @staticmethod
    def multiscale_discriminator(x, config, training, actv=tf.nn.leaky_relu, use_sigmoid=False, ksize=4, mode='real', reuse=False):
        """
        Function to build multiscale discriminator architecture for differentiating between real and reconstructed images 
        involving differentiating between images downsampled at different levels
        
        Input:
        x                : Input image
        config           : Configuration parameters
        training         : Variable to check for training phase
        actv             : Activation function for convolutional layers
        use_sigmoid      : Variable to check if vanilla GAN structure is to be used
        ksize            : Kernel size for convolutional layers
        mode             : Variable to specify whether discrimination operation on input image or reconstructed image
        reuse            : Reuse variable scope flag
        
        Output:
        out : Discriminator output at different downsampled images
        """  
        print('Building multiscale discriminator architecture')

        if mode == 'real':
            print('Building discriminator for {} [D(x)]'.format(mode))
        elif mode == 'reconstructed':
            print('Building discriminator for {} [D(G(z))]'.format(mode))
        else:
            raise NotImplementedError('Invalid discriminator mode specified.')

        # Downsample input for multiscale discriminator
        x2 = tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='same')
        x4 = tf.layers.average_pooling2d(x2, pool_size=3, strides=2, padding='same')

        print('Shape of x:', x.get_shape().as_list())
        print('Shape of x downsampled by factor 2:', x2.get_shape().as_list())
        print('Shape of x downsampled by factor 4:', x4.get_shape().as_list())

        def discriminator(x, scope, actv=actv, use_sigmoid=use_sigmoid, ksize=ksize, reuse=reuse):

            with tf.variable_scope('discriminator_{}'.format(scope), reuse=reuse):
                c1 = tf.layers.conv2d(x, 64, kernel_size=ksize, strides=2, padding='same', activation=actv)
                c2 = conv_block(c1, filters=128, kernel_size=ksize, strides=2, padding='same', actv=actv)
                c3 = conv_block(c2, filters=256, kernel_size=ksize, strides=2, padding='same', actv=actv)
                c4 = conv_block(c3, filters=512, kernel_size=ksize, strides=2, padding='same', actv=actv)
                out = tf.layers.conv2d(c4, 1, kernel_size=ksize, strides=1, padding='same')

                if use_sigmoid is True:  # Otherwise use LS-GAN
                    out = tf.nn.sigmoid(out)

            return out, c1, c2, c3, c4

        with tf.variable_scope('discriminator', reuse=reuse):
            disc, *Dk = discriminator(x, 'original')
            disc_downsampled_2, *Dk_2 = discriminator(x2, 'downsampled_2')
            disc_downsampled_4, *Dk_4 = discriminator(x4, 'downsampled_4')

        return disc, disc_downsampled_2, disc_downsampled_4, Dk, Dk_2, Dk_4


#============================================================================================================================
# Residual, upsample and convolutional blocks used in networks

def residual_block(x, n_filters, kernel_size=3, strides=1, actv=tf.nn.relu, training=True):

    init = tf.contrib.layers.xavier_initializer()
    strides = [1,1]
    identity_map = x

    p = int((kernel_size-1)/2)
    res = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
            activation=None, padding='VALID')
    res = actv(tf.contrib.layers.instance_norm(res))

    res = tf.pad(res, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
            activation=None, padding='VALID')
    res = tf.contrib.layers.instance_norm(res)

    assert res.get_shape().as_list() == identity_map.get_shape().as_list(), 'Mismatched shapes between input/output!'
    out = tf.add(res, identity_map)

    return out

def upsample_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu, batch_norm=False, training=True):

    bn_kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
    in_kwargs = {'center':True, 'scale': True}
    x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
    if batch_norm is True:
        x = tf.layers.batch_normalization(x, **bn_kwargs)
    else:
        x = tf.contrib.layers.instance_norm(x, **in_kwargs)
    x = actv(x)

    return x

def conv_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu):

    in_kwargs = {'center':True, 'scale': True}
    x = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
    x = tf.contrib.layers.instance_norm(x, **in_kwargs)
    x = actv(x)
    return x
