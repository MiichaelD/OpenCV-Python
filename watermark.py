import cv2
import numpy as np

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

# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 20, 255, cv2.THRESH_BINARY)
cv2.imshow('mask', mask)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('mask_inv', mask_inv)

def add_watermark(img): 
  roi = img[0:rows, 0:cols] # What's behind the watermark
  # Now black-out the area of logo in ROI
  img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv) # The black gets baked in
  cv2.imshow('img1_bg', img1_bg)

  # Take only region of logo from logo image.
  img2_fg = cv2.bitwise_and(img2, img2, mask = mask)
  cv2.imshow('img2_fg', img2_fg)

  # Put logo in ROI and modify the main image
  dst = cv2.add(img1_bg, img2_fg)
  dst = cv2.add(img1_bg, img2_fg)
  img[0:rows, 0:cols ] = dst
  cv2.imshow(window_name, img)
  return img

def show_camera():
  while cap.isOpened():
    success, img = cap.read()
    img = add_watermark(img)
    out.write(img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
      cap.release()
      out.release()
      print('Finished recording')
      cv2.imwrite('frame.jpg', img)
      break
  cv2.destroyAllWindows() 

show_camera()