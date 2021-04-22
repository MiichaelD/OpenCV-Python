import cv2
import numpy as np

window_name = 'Result'

img = cv2.imread("OpenCV-Python-Tutorials-and-Projects/Resources/lena.png")
cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FPS, 30)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('recording.avi', fourcc, fps, (width,height))

# Face recognition
faceCascade = cv2.CascadeClassifier(
  "OpenCV-Python-Tutorials-and-Projects/Intermediate/Custom Object Detection/haarcascades/haarcascade_frontalface_default.xml")

# mouse callback function
touches = []
def log_touches(event, x, y, flags, param):
  if event == 1:
    print('log_touches: ', event)
    touches.append((x,y))
    
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, log_touches)

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
  for t in touches:
    if len(faces) == 0:
      break
    f = faces[0]
    try:
      image[t[1]:t[1]+f[3], t[0]:t[0]+f[2]] = face
    except ValueError:
      pass
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