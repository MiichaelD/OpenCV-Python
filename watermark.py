import cv2
import numpy as np
from matplotlib import pyplot as plt

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
window_name = 'Result'

cap = cv2.VideoCapture(0)

pref_width = 1280
pref_height = 720
pref_fps = 30
cap.set(cv2.CAP_PROP_FPS, pref_fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('watermark.avi', fourcc, pref_fps, (width,height))
cv2.namedWindow(window_name)

# Load watermark image
img2 = cv2.imread('resources/opencv-logo.png')
# img2 = cv2.resize(img2, (pref_height//4, pref_width//4))

# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img2gray)
ret, mask = cv2.threshold(img2gray, 20, 255, cv2.THRESH_BINARY)
cv2.imshow('mask', mask)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('mask_inv', mask_inv)


plt.subplot(231),plt.imshow(img2[:,:,::-1],'gray'),plt.title('img2')
plt.subplot(232),plt.imshow(img2gray,'gray'),plt.title('img2gray')
plt.subplot(233),plt.imshow(mask,'gray'),plt.title('mask')
plt.subplot(234),plt.imshow(mask_inv,'gray'),plt.title('mask_inv')


def add_watermark(img, pos=None, size=None): 
  if pos is None:
    pos = (0, 0)
  if size is None:
    size = img2.shape
    mask1 = mask
    mask_inv1 = mask_inv
    img21 = img2
  else:
    mask1 = cv2.resize(mask, size, interpolation = cv2.INTER_LINEAR)
    mask_inv1 = cv2.resize(mask_inv, size, interpolation = cv2.INTER_LINEAR)
    img21 = cv2.resize(img2, size, interpolation = cv2.INTER_LINEAR)

  roi = img[pos[1]:pos[1]+size[0], pos[0]:pos[0]+size[1]] # What's behind the watermark
  # Now black-out the area of logo in ROI
  img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv1) # The black gets baked in
  cv2.imshow('img1_bg', img1_bg)

  # Take only region of logo from logo image.
  img2_fg = cv2.bitwise_and(img21, img21, mask = mask1)
  cv2.imshow('img2_fg', img2_fg)
  
  plt.subplot(235),plt.imshow(img1_bg[:,:,::-1],'gray'),plt.title('img1_bg')
  plt.subplot(236),plt.imshow(img2_fg[:,:,::-1],'gray'),plt.title('img2_fg')

  # Put logo in ROI and modify the main image
  dst = cv2.add(img1_bg, img2_fg)
  dst = cv2.add(img1_bg, img2_fg)
  img[pos[1]:pos[1]+size[0], pos[0]:pos[0]+size[1]] = dst
  cv2.imshow(window_name, img)
  return img

def show_camera():
  while cap.isOpened():
    success, img = cap.read()
    if not success:
      pass
    img = add_watermark(img, (200, 50))
    out.write(img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
      cap.release()
      out.release()
      print('Finished recording')
      cv2.imwrite('frame.jpg', img)
      break
  cv2.destroyAllWindows() 

# while (cv2.waitKey(0) & 0xFF) != ord('q'):
#   continue
# plt.show()
show_camera()