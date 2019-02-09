#!/usr/bin/python3
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

def generate_d5(path):

    """
    Implement Preprocessing if required
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

    start_time = time.time()
    G_loss_best, D_loss_best = float('inf'), float('inf')
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    # Load data
    print('Training on dataset')
    # if config.use_conditional_GAN:
    #     print('Using conditional GAN')
    #     paths, semantic_map_paths = Data.load_dataframe(directories.train, load_semantic_maps=True)
    #     test_paths, test_semantic_map_paths = Data.load_dataframe(directories.test, load_semantic_maps=True)
    # else:
    #     paths = Data.load_dataframe(directories.train)
    #     test_paths = Data.load_dataframe(directories.test)
    paths = Data.load_dataframe(directories.train)
    test_paths = Data.load_dataframe(directories.test)

    # Build graph
    model_gan = Model(config, paths, name=args.name)
    # GPUs = Utils.get_available_gpus()
    # if len(GPUs) >= 2:
    #     print("Using multi GPU model")
    #     gan = multi_gpu_model(model_gan, GPUs, cpu_relocation=True)
    # else:
    #     gan = model_gan
    gan = model_gan
    saver = tf.train.Saver()

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

        sess.run(gan.test_iterator.initializer, feed_dict=feed_dict_test_init)

        for epoch in range(config.num_epochs):

            sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_train_init)

            # Run diagnostics
            G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, directories, sess, saver, train_handle,
                start_time, epoch, args.name, G_loss_best, D_loss_best)
                    
            while True:
                try:
                    # Update generator
                    feed_dict = {gan.training_phase: True, gan.handle: train_handle}
                    summary,_=sess.run([gan.merge_op,gan.G_train_op], feed_dict=feed_dict)
                    gan.train_writer.add_summary(summary, epoch)

                    summary,step, _ = sess.run([gan.merge_op,gan.D_global_step, gan.D_train_op], feed_dict=feed_dict)
                    gan.train_writer.add_summary(summary, step)
                    G_loss_best, D_loss_best = Utils.run_diagnostics(gan, config, directories, sess, saver, train_handle, start_time, epoch, args.name, G_loss_best, D_loss_best)
                    if step % config.diagnostic_steps == 0:
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
                save_path = saver.save(sess, os.path.join(directories.checkpoints, '{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))

        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               '{}_end.ckpt'.format(args.name)),
                               global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-name", "--name", default="gan-train", help="Checkpoint/Tensorboard label")
    parser.add_argument("-path", "--path", default=None, help="Preprocessing Required",type=str)
    args = parser.parse_args()

    # Launch training
    if args.path:
        generate_d5(args.path)

    train(config_train, args)

if __name__ == '__main__':
    main()
