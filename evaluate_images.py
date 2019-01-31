import cv2
import sys, os


path = sys.argv[1]
abs_path = os.path.abspath(path)+'/'
file_names = os.listdir(path)
file_loc = [abs_path + x for x in file_names]
for file in file_loc:
	img = cv2.imread(file)
	print(img.shape)

	# fasdas