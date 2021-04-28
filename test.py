import cv2
import numpy as np

from goprocam import GoProCamera, constants


def init_gopro():
  gpCam = GoProCamera.GoPro()
  gpCam.stream("udp://127.0.0.1:10000", "high")
  # gpCam.stream("udp://127.0.0.1:10000.jpg", "high")
  # gpCam.stream("udp://127.0.0.1:10000.mpeg", "high")
  cap = cv2.VideoCapture("udp://127.0.0.1:10000")
  return cap

def init_webcam():
  frameWidth = 640
  frameHeight = 480
  cap = cv2.VideoCapture(0)
  cap.set(3, frameWidth)
  cap.set(4, frameHeight)
  cap.set(10,150)
  cap.set(cv2.CAP_PROP_FPS, 30)
  return cap


window_name = 'Test'
cap = init_webcam()
#cap = init_gopro()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(window_name+'.avi', fourcc, fps, (width,height))
cv2.namedWindow(window_name)

myColors = [[96, 168, 35, 165, 227, 255]
            # [5,107,0,19,255,255],
            # [133,56,0,159,156,255],
            # [57,76,0,100,255,255],
            # [90,48,0,118,255,255]
            ]
myColorValues = [[255,50,50]
                #  [51,153,255],          ## BGR
                #  [255,0,255],
                #  [0,255,0],
                #  [255,0,0]
                 ]

myPoints =  []  ## [x , y , colorId ]

def findColor(img,myColors,myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', imgHSV)
    count = 0
    newPoints=[]
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV,lower,upper)
        x, y = getContours(mask)
        if x != 0 and y != 0:
          newPoints.append([x,y,count])
        count +=1
        # cv2.imshow(str(color[0]), mask)
    return newPoints

def getContours(img):
    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
      area = cv2.contourArea(cnt)
      if area>500:
        cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, 0.1 * peri,True)
        x, y, w, h = cv2.boundingRect(approx)
    return x+w//2,y

def drawOnCanvas(myPoints, myColorValues):
  for point in myPoints:
    cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)

print('Starting livestream')
errors = 10
while True:
  success, img = cap.read()
  if not success: 
    print('Error while reading camera input')
    errors -= 1
    if errors < 1:
      break
    continue
  imgResult = img.copy()
  newPoints = findColor(img, myColors,myColorValues)
  if len(newPoints)!=0:
    for newP in newPoints:
      myPoints.append(newP)
  if len(myPoints)!=0:
    drawOnCanvas(myPoints, myColorValues)
  cv2.imshow(window_name, imgResult)
  # out.write(imgResult)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cap.release()
    # out.release()
    break
  cv2.destroyAllWindows() 