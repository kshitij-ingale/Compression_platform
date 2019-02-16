#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import directories, image_properties

class Data(object):

    @staticmethod
    def load_dataframe(filename, load_semantic_maps=False):

        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)
        return df['path'].values

    @staticmethod
    def load_dataset(image_paths, batch_size, test=False, **kwargs):

        def _parser(image_path, semantic_map_path=None):
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


    @staticmethod
    def load_inference(filenames, labels, batch_size, resize=(32,32)):

        # Single image estimation over multiple stochastic forward passes

        def _preprocess_inference(image_path, label, resize=(32,32)):
            # Preprocess individual images during inference
            image_path = tf.squeeze(image_path)
            image = tf.image.decode_png(tf.read_file(image_path))
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.image.resize_images(image, size=resize)

            return image, label

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(_preprocess_inference)
        dataset = dataset.batch(batch_size)
        
        return dataset

