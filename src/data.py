#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import directories, image_properties

class Data(object):

    @staticmethod
    def load_dataframe(filename, load_semantic_maps=False):

        df = pd.read_hdf(filename, key='df').sample(frac=0.03).reset_index(drop=True)
        return df['path'].values

    @staticmethod
    def load_dataset(image_paths, batch_size, test=False, augment=False, downsample=False, **kwargs):

        # def _augment(image):
        #     # On-the-fly data augmentation
        #     image = tf.image.random_brightness(image, max_delta=0.1)
        #     image = tf.image.random_contrast(image, 0.5, 1.5)
        #     image = tf.image.random_flip_left_right(image)

        #     return image

        def _parser(image_path, semantic_map_path=None):

            # def _aspect_preserving_width_resize(image, width=512):
            #     height_i = tf.shape(image)[0]
            #     # width_i = tf.shape(image)[1]
            #     # ratio = tf.to_float(width_i) / tf.to_float(height_i)
            #     # new_height = tf.to_int32(tf.to_float(height_i) / ratio)
            #     new_height = height_i - tf.floormod(height_i, 16)
            #     return tf.image.resize_image_with_crop_or_pad(image, new_height, width)

            # def _image_decoder(path):

                # if training_dataset == 'faces':
                #     im = tf.image.decode_jpeg(tf.read_file(path), channels=image_properties.depth)
                # else:
                #     im = tf.image.decode_png(tf.read_file(path), channels=3)
                # im = tf.image.decode_image(tf.read_file(path), channels=image_properties.depth)
                # im = tf.image.convert_image_dtype(im, dtype=tf.float32)
                # return 2 * im - 1 # [0,1] -> [-1,1] (tanh range)
                    
            # image = _image_decoder(image_path)
            
            # size = tf.constant([image_properties.height, image_properties.width, image_properties.depth])
            
            im = tf.image.decode_image(tf.read_file(image_path), channels=image_properties.depth)
            im = tf.image.convert_image_dtype(im, dtype=tf.float32)
            im = 2 * im - 1 # [0,1] -> [-1,1] (tanh range)
            
            # image = _aspect_preserving_width_resize(im)
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

