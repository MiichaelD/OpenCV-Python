import cv2
import numpy as np
from goprocam import GoProCamera, constants

gpCam = GoProCamera.GoPro()
# gpCam.gpControlSet(constants.Stream.BIT_RATE, constants.Stream.BitRate.B2_4Mbps)
# gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R480)
cap = cv2.VideoCapture("udp://127.0.0.1:10000", cv2.CAP_FFMPEG)
while True:
  # Capture frame-by-frame
  ret, frame = cap.read()
  if not ret:
    print('Capture read failed')
    pass
  # Display the resulting frame
  cv2.imshow("GoPro OpenCV", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.imwrite('gopro_frame.jpg', frame)
    break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()