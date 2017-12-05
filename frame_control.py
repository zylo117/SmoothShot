import imutils
from imutils.video.videostream import VideoStream
import cv2
import datetime
import time
import numpy as np

# if a video path was not supplied, grab the reference to the webcam
camera = VideoStream(src=0).start()

current_fps = 0

# keep looping
count = 0
old_frame = None
while True:
    frame = camera.read()

    if frame is old_frame:
        continue

    if count % 2 == 0:
        # 帧率计数
        insert_frame = frame
        start_time = datetime.datetime.now()

    if count % 2 == 1:
        # 帧率计数
        end_time = datetime.datetime.now()
        two_frame_time = (end_time - start_time).total_seconds()
        current_fps = (1 / (two_frame_time / 2))
    cv2.putText(frame, str(int(current_fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 出图
    cv2.imshow("Frame", frame)
    count += 1

    # 播放插入帧
    # if count > 1:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # 更新数据
    old_frame = frame

camera.stop()
cv2.destroyAllWindows()

