import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat

vc = cv2.VideoCapture(1)
window_name = 'Result'
if not vc.isOpened():
    raise RuntimeError('Could not open video source')

# Face recognition
faceCascade = cv2.CascadeClassifier(
  "OpenCV-Python-Tutorials-and-Projects/Intermediate/Custom Object Detection/haarcascades/haarcascade_frontalface_default.xml")

def show_faces(image, flip=False):
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
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
  cv2.putText(image, (str(fps) + ' FPS'), (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
  cv2.imshow(window_name, image)
  return image

pref_width = 1280
pref_height = 720
pref_fps = 30
vc.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
vc.set(cv2.CAP_PROP_FPS, pref_fps)

# Query final capture device values
# (may be different from preferred settings)
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vc.get(cv2.CAP_PROP_FPS))

with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR) as cam:
  print('Virtual camera device: ' + cam.device)
  while True:
    ret, frame = vc.read()
    if not ret:
      continue
    # .. apply your filter ..
    frame = show_faces(frame)
    # .. then continue ..
    cam.send(frame)
    cam.sleep_until_next_frame()
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
      vc.release()
      cv2.imwrite('output/virtual_camera_last_frame.jpg', frame)
      print('Finished virtual camera output. Frames sent:', cam.frames_sent)
      break
  cv2.destroyAllWindows() 


