#!/usr/bin/python3
# Script for preprocessing input image

import cv2
import os
import argparse


from config import input_attributes

def main():
    """
    Function to parse arguments and run inference

    Input:
    Parse arguments using argparse

    Output:
    None (Preprocessed images are replaced with original images)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="path to input images directory", type=str)
    parser.add_argument("-o", "--output", help="path to output images directory", type=str)
    args = parser.parse_args()

    in_abs_path = os.path.abspath(args.input)+'/'
    out_abs_path = os.path.abspath(args.output)+'/'
    file_names = os.listdir(args.input)
    all_file_loc = [in_abs_path + x for x in file_names]

    for ct,file in enumerate(all_file_loc):

        image = cv2.imread(file,cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (input_attributes.WIDTH, input_attributes.HEIGHT))
        # Replace original images (Change output directory if required)
        output = out_abs_path + file_names[ct]
        cv2.imwrite(output,resized_image)

if __name__ == '__main__':
    main()