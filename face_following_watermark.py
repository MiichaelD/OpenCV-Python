import cv2
import numpy as np

window_name = 'Result'

img = cv2.imread("OpenCV-Python-Tutorials-and-Projects/Resources/lena.png")
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 30)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('face_following_watermark.avi', fourcc, fps, (width,height))
cv2.namedWindow(window_name)

# Face recognition
faceCascade = cv2.CascadeClassifier(
  "OpenCV-Python-Tutorials-and-Projects/Intermediate/Custom Object Detection/haarcascades/haarcascade_frontalface_default.xml")

# Load watermark image
img2 = cv2.imread('resources/opencv-logo.png')
rows, cols, channels = img2.shape
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 20, 255, cv2.THRESH_BINARY)
cv2.imshow('mask', mask)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('mask_inv', mask_inv)

def add_watermark(img, pos=None, size=None): 
  if pos is None:
    pos = (0, 0)
  if size is None:
    size = (rows, cols)
    mask1 = mask
    mask_inv1 = mask_inv
    img21 = img2
  else:
    mask1 = cv2.resize(mask, size, interpolation = cv2.INTER_LINEAR)
    mask_inv1 = cv2.resize(mask_inv, size, interpolation = cv2.INTER_LINEAR)
    img21 = cv2.resize(img2, size, interpolation = cv2.INTER_LINEAR)

  roi = img[pos[1]:pos[1]+size[1], pos[0]:pos[0]+size[0]] # What's behind the watermark
  # Now black-out the area of logo in ROI
  img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv1) # The black gets baked in
  cv2.imshow('img1_bg', img1_bg)

  # Take only region of logo from logo image.
  img2_fg = cv2.bitwise_and(img21, img21, mask = mask1)
  cv2.imshow('img2_fg', img2_fg)

  # Put logo in ROI and modify the main image
  dst = cv2.add(img1_bg, img2_fg)
  dst = cv2.add(img1_bg, img2_fg)
  img[pos[1]:pos[1]+size[1], pos[0]:pos[0]+size[0]] = dst
  cv2.imshow(window_name, img)
  return img

def show_faces(image=None, flip=False):
  if image is None:
    image = img
  if flip:
    image = cv2.flip(image, 1)
  imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(
        imgGray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
  face = None
  for (x, y, w, h) in faces:
    face = image[y:y+w, x:x+h]
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    add_watermark(image, (x, y), (w, h))
  cv2.putText(image, (str(fps) + ' FPS'), (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
  out.write(image)
  cv2.imshow(window_name, image)
  return image

def show_camera():
  while cap.isOpened():
    global img
    success, img = cap.read()
    img = show_faces(img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
      cap.release()
      out.release()
      print('Finished recording')
      cv2.imwrite('frame.jpg', img)
      break
  cv2.destroyAllWindows() 

show_camera()