#!/usr/bin/python3

# Script to run inference (i.e. image encoding and decoding) on a single image.
# Code borrowed from Justin Tan (https://github.com/Justin-Tan/generative-compression) and modified as required

import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_test, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def generate_df(path):
    """
    Function to obtain hdf file given the input images directory

    Input:
    path : Input image directory

    Output:
    None (File saved in directory as per config file)
    """
    if path.endswith('jpg'):
        return np.array([os.path.abspath(path)])
    abs_path = os.path.abspath(path)+'/'
    file_names = os.listdir(path)
    file_loc = [abs_path + x for x in file_names]
    # test = pd.DataFrame({'path':file_loc})
    # test.to_hdf(directories.infer, 'df', table=True, mode='a')

    return np.array(file_loc)


def infer_compress(config, args):
    """
    Function to run inference and compress input image

    Input:
    config : Configuration parameters as defined in config file
    args   : Input arguments as parsed by argparse

    Output:
    None (File saved in directory as per config file)
    """
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    assert (ckpt.model_checkpoint_path), 'Provide checkpoint file for inference'

    # test_paths = Data.load_dataframe(directories.infer)
    test_paths = generate_df(args.path)
    comp_model = Model(config, test_paths, name='infer_compress', evaluate=True)
    saver = tf.train.Saver()
    feed_dict_init = {comp_model.test_path_placeholder: test_paths}

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        handle = sess.run(comp_model.test_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))

        sess.run(comp_model.test_iterator.initializer, feed_dict=feed_dict_init)
        i = 0
        if args.path.endswith('jpg'):
            file_names = [args.path]
        else:
            file_names = os.listdir(args.path)
        while True:
            try:
                print("Currently processing {}".format(file_names[i][:-4]))
                eval_dict = {comp_model.training_phase: False, comp_model.handle: handle}

                if args.output_path is None:
                    output = os.path.splitext(os.path.basename(args.path))
                    save_path = os.path.join(directories.samples, '{}_compressed'.format(file_names[i][:-4]))
                else:
                    save_path = args.output_path
                Utils.single_plot(0, 0, sess, comp_model, handle, save_path, config, single_compress=True)
                i += 1

            except (tf.errors.OutOfRangeError, IndexError):
                print('Inference completed')
                break
    return


def infer_decompress(config, args):
    """
    Function to run inference and reconstruct image after decoding the compressed vector

    Input:
    config : Configuration parameters as defined in config file
    args   : Input arguments as parsed by argparse

    Output:
    None (File saved in directory as per config file)
    """
    
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    assert (ckpt.model_checkpoint_path), 'Provide checkpoint file for inference'

    paths = np.array([args.compressed_path])
    comp_model = Model(config, paths, name='infer_decompress', evaluate=True)
    saver = tf.train.Saver()
    
    feed_dict_init = {comp_model.path_placeholder: paths}
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        handle = sess.run(comp_model.train_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))
        
        sess.run(comp_model.train_iterator.initializer, feed_dict=feed_dict_init)
        
        eval_dict = {comp_model.training_phase: False, comp_model.handle: handle}
        
        assert( args.compressed_path is not None), 'Input has not been specified'
        
        if args.output_path is None:
            output = os.path.splitext(os.path.basename(args.image_path))
            save_path = os.path.join(directories.samples, '{}_compressed.pdf'.format(output[0]))
        else:
            save_path = args.output_path
        Utils.decode(sess, comp_model, handle, args.compressed_path, save_path, config)

        print('Reconstructed image saved to', save_path)

    return

def main(**kwargs):
    """
    Function to parse arguments and run inference

    Input:
    Input arguments as parsed by argparse

    Output:
    None (Run inference by calling appropriate function)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-c", "--compressed_path", help="path to compressed file", type=str)
    parser.add_argument("-o", "--output_path", help="path to output image", type=str)
    parser.add_argument("-path", "--path", default=None, help="Directory to multiple input images",type=str)
    args = parser.parse_args()

    # Launch training
    args = parser.parse_args()

    if args.path:
        infer_compress(config_test, args)
    if args.compressed_path:
        infer_decompress(config_test, args)

if __name__ == '__main__':
    main()
