import cv2
import sys, os
import random
from config import image_properties
import pandas as pd
import matplotlib.pyplot as plt
path = sys.argv[1]
abs_path = os.path.abspath(path)+'/'
file_names = os.listdir(path)
all_file_loc = [abs_path + x for x in file_names]
s1=[]
s2=[]
for ct,file in enumerate(all_file_loc):
	image = cv2.imread(file,cv2.IMREAD_COLOR)
#	s1.append(image.shape[0])
#	s2.append(image.shape[1])
#	resized_image = cv2.resize(image, (image_properties.width, image_properties.height)
	resized_image = image[:,1:-1]
	cv2.imwrite(file,resized_image)
	print(ct)

# size_df = pd.DataFrame({'shape_1':s1,'shape_2':s2})
# size_df.hist()
# plt.show()
