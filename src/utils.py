# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session

import tensorflow as tf
import numpy as np
import os, time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import struct
from array import array

from config import directories, image_properties

class Utils(object):
    
    @staticmethod
    def conv_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu):
        in_kwargs = {'center':True, 'scale': True}
        x = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
        x = tf.contrib.layers.instance_norm(x, **in_kwargs)
        x = actv(x)

        return x

    @staticmethod
    def upsample_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu):
        in_kwargs = {'center':True, 'scale': True}
        x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
        x = tf.contrib.layers.instance_norm(x, **in_kwargs)
        x = actv(x)

        return x

    @staticmethod
    def residual_block(x, n_filters, kernel_size=3, strides=1, actv=tf.nn.relu):
        init = tf.contrib.layers.xavier_initializer()
        # kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
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

    @staticmethod
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        GPU_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
        print(GPU_list)
        return GPU_list

    @staticmethod
    def scope_variables(name):
        with tf.variable_scope(name):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    @staticmethod
    def run_diagnostics(model, config, directories, sess, saver, train_handle, start_time, epoch, name, G_loss_best, D_loss_best):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_test = {model.training_phase: False, model.handle: train_handle}

        try:
            G_loss, D_loss, summary = sess.run([model.G_loss, model.D_loss, model.merge_op], feed_dict=feed_dict_test)
            model.train_writer.add_summary(summary)
        except tf.errors.OutOfRangeError:
            G_loss, D_loss = float('nan'), float('nan')

        if G_loss < G_loss_best and D_loss < D_loss_best:
            G_loss_best, D_loss_best = G_loss, D_loss
            improved = '[*]'
            if epoch>5:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, '{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))

        if epoch % 5 == 0 and epoch > 5:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, '{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))

        print('Epoch {} | Generator Loss: {:.3f} | Discriminator Loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, G_loss, D_loss, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

        return G_loss_best, D_loss_best

    @staticmethod
    def single_plot(epoch, global_step, sess, model, handle, name, config, single_compress=False):

        real = model.example[0]
        gen = model.reconstruction[0]
        quantized_z = model.z
        # Generate images from noise, using the generator network.
        r, g, z = sess.run([real, gen, quantized_z], feed_dict={model.training_phase:True, model.handle: handle})
        
        images = list()

        for im, imtype in zip([r,g], ['real', 'gen']):
            im = ((im+1.0))/2  # [-1,1] -> [0,1]
            im = np.squeeze(im)
            if image_properties.depth == 1:
                im = im[:,:,]
            else:
                im = im[:,:,:3]
            images.append(im)

            # Uncomment to plot real and generated samples separately
            f = plt.figure()
            plt.imshow(im)
            plt.axis('off')
            if single_compress:
                #f.savefig(name+'.jpg', format = 'jpg', dpi=720, bbox_inches='tight', pad_inches=0)
                plt.imsave(name+'.jpg',np.asarray(im))
                plt.gcf().clear()
            plt.close(f)
        comparison = np.hstack(images)
        f = plt.figure()
        plt.imshow(comparison)
        plt.axis('off')
        if single_compress:
            f.savefig(name, format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)            
            write_compressed_file(z,name)
        else:
            f.savefig("{}/gan_compression_{}_epoch{}_step{}_{}_comparison.pdf".format(directories.samples, name, epoch,
                global_step, imtype), format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
        plt.gcf().clear()
        plt.close(f)

    @staticmethod
    def decode( sess, model, handle, input, name, config):
        recon = model.reconstruction
        # this needs to be loaded
        quantized_z = read_compressed_file(input)
        # Generate images from noise, using the generator network.
        g = sess.run( recon, feed_dict={model.w_hat:quantized_z, model.training_phase:True, model.handle: handle})
        # now plot the image
        im = ((g+1.0))/2  # [-1,1] -> [0,1]
        im = np.squeeze(im)
        if image_properties.depth == 1:
            im = im[:,:,]
        else:
            im = im[:,:,:3]

        f = plt.figure()
        plt.imshow(im)
        plt.axis('off')
        plt.imsave(name+'.jpg',np.asarray(im))
        plt.gcf().clear()
        plt.close(f)

    @staticmethod
    def weight_decay(weight_decay, var_label='DW'):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'{}'.format(var_label)) > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(weight_decay, tf.add_n(costs))

def write_compressed_file(np_array, out_file = 'compressed_x'):

    bitstring = ''            
    for center in np_array.astype(int).flatten():
        if center == 0:
            bitstring+='00'
        elif center == 1:
            bitstring+='10'
        elif center == 2:
            bitstring+='01'
        else:
            print('Something went wrong' )
        
    print('File size = ',len(bitstring))
    with open(out_file+'.txt', 'w') as f:
        f.write(bitstring)
        #These blocks used to save file to bytes, come back to this lates
#    bin_array = array("B")

#    for index in range(0, len(bitstring), 8):
#        byte = bitstring[index:index + 8][::-1]
#        bin_array.append(int(byte, 2))
        
#    with open(out_file+'.bin', 'wb') as f:
#        for b in bin_array:
#            f.write(struct.pack('h', b))
    #np_array = tf.Session().run(tf_array)
    #with h5py.File(out_file+'.h5', 'w') as hf:
    #    hf.create_dataset("quantized_image",  data=np_array)

def read_compressed_file(input_file):
    with open(input_file, 'r') as f:
        data=str(f.readlines())
    f.close()
    length = len(data)-2
    im_list = []
    for ii in range(1,length//2):
        nn = data[2*ii:2*ii+2]
        if nn == '00':
            im_list.append(0.)
        elif nn == '10':
            im_list.append(1.)
        elif nn == '01':
            im_list.append(2.)
    im_mat = np.array(im_list)
    return im_mat.reshape(image_properties.compressed_dims)
