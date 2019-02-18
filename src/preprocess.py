#!/usr/bin/python3
# Script for preprocessing input image

import cv2
import sys, os

from config import image_properties

def main(path):
    """
    Function to parse arguments and run inference

    Input:
    path : Directory to images that need to be preprocessed

    Output:
    None (Preprocessed images are replaced with original images)
    """

	abs_path = os.path.abspath(path)+'/'
	file_names = os.listdir(path)
	all_file_loc = [abs_path + x for x in file_names]

	for ct,file in enumerate(all_file_loc):
		image = cv2.imread(file,cv2.IMREAD_COLOR)
		resized_image = cv2.resize(image, (image_properties.WIDTH, image_properties.HEIGHT))
		# Replace original images (Change output directory if required)
		output = file
		cv2.imwrite(output,resized_image)

if __name__ == '__main__':
    main(sys.argv[1])