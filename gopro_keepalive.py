# https://github.com/KonradIT/gopro-py-api/blob/master/examples/opencv_gopro/gopro_keepalive.py
from goprocam import GoProCamera, constants

gopro = GoProCamera.GoPro()
gopro.stream("udp://127.0.0.1:10000")