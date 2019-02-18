# -*- coding: utf-8 -*-
# Script for utility functions like comparing current model parameters, reading or writing quantized vectors
# Code borrowed from Justin Tan (https://github.com/Justin-Tan/generative-compression) and modified as required


import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from config import directories, image_properties

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
        
        r, g, z = sess.run([real, gen, quantized_z], feed_dict={model.training_phase:True, model.handle: handle})
        
        images = []
        for im in [r,g]:
            im = ((im+1.0))/2  # [-1,1] -> [0,1]
            im = np.squeeze(im)
            if image_properties.DEPTH == 1:
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
            f.savefig(name, format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)            
            write_compressed_file(z,name)
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
        quantized_z = read_compressed_file(input)
        # Obtain reconstructed image 
        g = sess.run( recon, feed_dict={model.z:quantized_z, model.training_phase:True, model.handle: handle})

        # Convert from [-1,1] domain to [0,1]
        im = ((g+1.0))/2
        im = np.squeeze(im)
        if image_properties.DEPTH == 1:
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

def write_compressed_file(data, out_file = 'compressed_x'):
    """
    Function to write quantized vector of image as a binary file
    
    Input:
    data : Quantized vector of image
    out_file : File name for binary file to be stored
    
    Output:
    None (File saved in location)
    """
    
    bitstring = ''            
    for center in data.astype(int).flatten():
        if center == 0:
            bitstring+='00'
        elif center == 1:
            bitstring+='10'
        elif center == 2:
            bitstring+='01'
        elif center == 3:
            bitstring += '11'
        else:
            print('Error in encoding' )
        
    print('File size = ',len(bitstring))
    with open(out_file+'.bin', 'w') as f:
        f.write(bitstring)

def read_compressed_file(input_file):
    """
    Function to write quantized vector of image as a binary file
    
    Input:
    input_file : File name for binary file to be restored to reconstructed image
    
    Output:
    Matrix containing quantized vector for reconstruction
    """

    with open(input_file, 'rb') as f:
        data=str(f.read())
    f.close()

    comp = []
    for i in range(0,len(data),2):
        bit = data[i:i+2]
        if bit == '00':
            comp.append(0.)
        elif bit == '10':
            comp.append(1.)
        elif bit == '01':
            comp.append(2.)
        elif bit == '11':
            comp.append(3.)
    im_mat = np.array(comp)
    return im_mat.reshape(image_properties.compressed_dims)
