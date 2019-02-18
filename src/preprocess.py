#!/usr/bin/python3
# Script for preprocessing input image

import cv2
import sys, os

from config import image_properties

# Preprocess image to resize image as per dimensions specified in user defined config file
path = sys.argv[1]
abs_path = os.path.abspath(path)+'/'
file_names = os.listdir(path)
all_file_loc = [abs_path + x for x in file_names]

for ct,file in enumerate(all_file_loc):
	image = cv2.imread(file,cv2.IMREAD_COLOR)
	resized_image = cv2.resize(image, (image_properties.width, image_properties.height))
	cv2.imwrite(file,resized_image)