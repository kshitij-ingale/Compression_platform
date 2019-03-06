# -*- coding: utf-8 -*-
# Script for utility functions like comparing current model parameters, reading or writing quantized vectors
# Code borrowed from Justin Tan (https://github.com/Justin-Tan/generative-compression) and modified as required


import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from config import directories, input_attributes
from arithmetic_encoder import arithmetic_encoder

class Utils(object):
    
    @staticmethod
    def run_diagnostics(model, config, sess, saver, train_handle, start_time, epoch, name, G_loss_best, D_loss_best):
        """
        Function to evaluate model performance and save checkpoint 

        Input:
        model        : Current step model instance
        config       : Configuration parameters
        sess         : Tensorflow session instance
        train_handle : train handle string 
        start_time   : Starting time for the current epoch
        epoch        : Current epoch
        name         : Model name
        G_loss_best  : Lowest loss value for generator
        D_loss_best  : Lowest loss value for discriminator
        
        Output:
        G_loss_best  : Current lowest loss value for generator
        D_loss_best  : Current lowest loss value for discriminator
        """

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
                save_path = saver.save(sess, os.path.join(directories.checkpoints_best, '{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
                print('Current Best Graph saved to file: {}'.format(save_path))

        print('Epoch {} | Generator Loss: {:.3f} | Discriminator Loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, G_loss, D_loss, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

        return G_loss_best, D_loss_best

    @staticmethod
    def single_plot(epoch, global_step, sess, model, handle, name, config, single_compress=False):
        """
        Function to obtain reconstructed image and compare with input image

        Input:
        epoch           : Current epoch
        global_step     : Current step
        sess            : Tensorflow session instance
        model           : Current model instance
        handle          : Handle string corresponding to train/test
        name            : File name to be saved
        config          : Configuration parameters
        single_compress : Variable to check if inference on single file or evaluation step during training
        
        Output:
        None (File saved in location)
        """

        real = model.example[0]
        gen = model.reconstruction[0]
        quantized_z = model.z

        # t0 = time.time()
        r, g, z = sess.run([real, gen, quantized_z], feed_dict={model.training_phase:True, model.handle: handle})
        # print("Time required for inference is {}".format(time.time()-t0))
        
        images = []
        for im in [r,g]:
            im = ((im+1.0))/2  # [-1,1] -> [0,1]
            im = np.squeeze(im)
            if input_attributes.DEPTH == 1:
                im = im[:,:,]
            else:
                im = im[:,:,:3]
            images.append(im)

            f = plt.figure()
            plt.imshow(im)
            plt.axis('off')
            if single_compress:
                plt.imsave(name+'.jpg',np.asarray(im))
                plt.gcf().clear()
            plt.close(f)
        comparison = np.hstack(images)
        f = plt.figure()
        plt.imshow(comparison)
        plt.axis('off')
        if single_compress:
            f.savefig(name+'comparison', format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)            
            # write_compressed_file(z,name)
            arithmetic_encoder.compress(z,name)
        else:
            f.savefig("{}/gan_compression_{}_epoch{}_step{}_comparison.pdf".format(directories.samples, name, epoch,
                global_step), format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
        plt.gcf().clear()
        plt.close(f)

    @staticmethod
    def decode(sess, model, handle, input, name, config):
        """
        Function to obtain reconstructed image from quantized vector 
        
        Input:
        sess         : Tensorflow session instance
        model        : Current step model instance
        handle       : Handle string corresponding to train/test
        input        : Input file consisting of compressed quantized vector
        name         : Model name
        config       : Configuration parameters
        
        Output:
        None (File saved in location)
        """
        # reconstruction module from model
        recon = model.reconstruction
        # compressed vector as obtained from input file
        quantized_z = arithmetic_encoder.decompress(input)
        quantized_z = quantized_z.reshape(input_attributes.compressed_dims)
        # Obtain reconstructed image 
        g = sess.run(recon, feed_dict={model.z:quantized_z, model.training_phase:True, model.handle: handle})

        # Convert from [-1,1] domain to [0,1]
        im = ((g+1.0))/2
        im = np.squeeze(im)
        if input_attributes.DEPTH == 1:
            im = im[:,:,]
        else:
            im = im[:,:,:3]
        # Save reconstructed image
        f = plt.figure()
        plt.imshow(im)
        plt.axis('off')
        plt.imsave(name+'.jpg',np.asarray(im))
        plt.gcf().clear()
        plt.close(f)



