import numpy 
import math
import cv2
import sys
from skimage.measure import compare_ssim as ssim


original = cv2.imread(sys.argv[1])
compress = cv2.imread(sys.argv[2],1)
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
    	return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

ssim_ = ssim(original, compress)
d=psnr(original,compress)
print("PSNR is ",d)
print("SSIM is ",ssim_)
