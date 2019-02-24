#!/usr/bin/python3
# Script to train the GAN based image compression model
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
from config import config_train, directories

from tensorflow.keras.utils import multi_gpu_model


tf.logging.set_verbosity(tf.logging.ERROR)

def generate_hdf(path):
    """
    Function to obtain hdf file given the input images directory

    Input:
    path : Input image directory

    Output:
    None (File saved in directory as per config file)
    """

    abs_path = os.path.abspath(path)+'/'
    file_names = os.listdir(path)

    if len(file_names)%config_train.batch_size!=0:
        while len(file_names)%config_train.batch_size!=0:
            file_names = file_names[:-1]
    
    file_loc = [abs_path + x for x in file_names]
    train = pd.DataFrame({'path':file_loc[:int(len(file_names)*config_train.train_fraction)]})
    test = pd.DataFrame({'path':file_loc[int(len(file_names)*config_train.train_fraction):]})
    train.to_hdf(directories.train, 'df', table=True, mode='a')
    test.to_hdf(directories.test, 'df', table=True, mode='a')


def train(config, args):
    """
    Function to instantiate and train model

    Input:
    config : Configuration parameters as defined in config file
    args   : Input arguments as parsed by argparse

    Output:
    None (File saved in directory as per config file)
    """

    start_time = time.time()
    G_loss_best, D_loss_best = float('inf'), float('inf')
    # Load checkpoint
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    # Load training and test dataset
    print('Training on dataset')
    paths = Data.load_dataframe(directories.train)
    test_paths = Data.load_dataframe(directories.test)

    # Build computational graph of the model through model instance
    gan = Model(config, paths, name=args.name)
    
    saver = tf.train.Saver()
    # Generate feed dictionary for Tensorflow session
    feed_dict_test_init = {gan.test_path_placeholder: test_paths}
    feed_dict_train_init = {gan.path_placeholder: paths}

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_handle = sess.run(gan.train_iterator.string_handle())
        test_handle = sess.run(gan.test_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            # Continue training saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))

        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('{} restored.'.format(args.restore_path))

        # sess.run(gan.test_iterator.initializer, feed_dict=feed_dict_test_init)

        for epoch in range(config.num_epochs):

            sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_train_init)

            # Evaluate model performance
            G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, sess, saver, train_handle, start_time, epoch, args.name, G_loss_best, D_loss_best)
                    
            while True:
                try:
                    # Update generator
                    feed_dict = {gan.training_phase: True, gan.handle: train_handle}
                    summary,_=sess.run([gan.merge_op,gan.G_train_op], feed_dict=feed_dict)
                    gan.train_writer.add_summary(summary, epoch)

                    summary,step, _ = sess.run([gan.merge_op,gan.D_global_step, gan.D_train_op], feed_dict=feed_dict)
                    gan.train_writer.add_summary(summary, step)
                    
                    # Evaluate model performance during an epoch after steps specified in config file
                    if step % config.diagnostic_steps == 0:
                        G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, directories, sess, saver, train_handle, start_time, epoch, args.name, G_loss_best, D_loss_best)
                        Utils.single_plot(epoch, step, sess, gan, train_handle, args.name, config)
                        

                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        '{}_last.ckpt'.format(args.name)), global_step=epoch)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()
                    
            if epoch % 5 == 0 and epoch > 5:
                save_path = saver.save(sess, os.path.join(directories.checkpoints, '{}_epoch{}.ckpt'.format(args.name, epoch)), global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))

        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               '{}_end.ckpt'.format(args.name)),
                               global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

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
    parser.add_argument("-name", "--name", default="gan-train", help="Checkpoint/Tensorboard label")
    parser.add_argument("-path", "--path", default=None, help="Directory to input images",type=str)
    args = parser.parse_args()

    if args.path:
        generate_hdf(args.path)

    train(config_train, args)

if __name__ == '__main__':
    main()
