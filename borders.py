import cv2
import numpy as np
from matplotlib import pyplot as plt

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
BLUE = [255,0,0]
window_name = 'Result'

img = cv2.imread("OpenCV-Python-Tutorials-and-Projects/Resources/lena.png")
img1 = img[:,:,::-1]

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.xticks([]), plt.yticks([]),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.xticks([]), plt.yticks([]),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.xticks([]), plt.yticks([]),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.xticks([]), plt.yticks([]),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()