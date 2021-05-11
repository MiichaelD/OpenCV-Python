# https://github.com/KonradIT/gopro-py-api/blob/master/examples/opencv_gopro/gopro_keepalive.py
from goprocam import GoProCamera, constants

gopro = GoProCamera.GoPro()
# gopro.video_settings(res='1080p', fps='30')
# gopro.gpControlSet(constants.Stream.WINDOW_SIZE, constants. Stream.WindowSize.R720)
gopro.stream("udp://127.0.0.1:10000")