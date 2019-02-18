#!/usr/bin/python3
# Script to read input dataset
# Code borrowed from Justin Tan (https://github.com/Justin-Tan/generative-compression) and modified as required

import tensorflow as tf
import pandas as pd
from config import image_properties

class Data(object):

    @staticmethod
    def load_dataframe(filename):
        """
        Function to build encoder architecture for encoding image (H,W,d) to latent feature map (H/16,W/16,C)
        
        Input:
        filename : Input hdf5 file consisting of training dataset
        
        Output:
        dataframe of paths to images dataset
        """

        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)
        return df['path'].values

    @staticmethod
    def load_dataset(image_paths, batch_size, test=False, **kwargs):
        """
        Function to build encoder architecture for encoding image (H,W,d) to latent feature map (H/16,W/16,C)
        
        Input:
        image_paths : Paths to input images
        batch_size  : Batch size as per user defined config file
        test        : Variable to check if test dataset
        
        Output:
        dataset : Tensorflow dataset instance of the training/test dataset
        """

        def _parser(image_path, semantic_map_path=None):
            # parser function for dataset instance mapping from input dataframe consisting of image path to image tensor
            im = tf.image.decode_image(tf.read_file(image_path), channels=image_properties.depth)
            im = tf.image.convert_image_dtype(im, dtype=tf.float32)
            im = 2 * im - 1 # [0,1] -> [-1,1] (tanh range)
            im.set_shape([image_properties.height,image_properties.width,image_properties.depth])
            return im

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(_parser)
        dataset = dataset.shuffle(buffer_size=8)
        dataset = dataset.batch(batch_size)

        if test:
            dataset = dataset.repeat()

        return dataset