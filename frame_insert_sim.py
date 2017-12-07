import imutils
from imutils.video.videostream import VideoStream
import cv2
from imutils.video import FPS
import datetime
import numpy as np

# if a video path was not supplied, grab the reference to the webcam
camera = VideoStream(src=0).start()

fps = FPS().start()
current_fps = 0

# keep looping
old_centroid = [0, 0]
while True:
    frame = camera.read()

    if fps._numFrames % 2 == 0:
        # 帧率计数
        start_time = datetime.datetime.now()

    frame = imutils.resize(frame, width=400)

    # 生成插入帧
    if fps._numFrames % 2 == 0:
        insert_frame = frame

    # 计算帧间重心Delta矢量
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV, dst=blurred)
    M = cv2.moments(blurred)  # 计算矩
    cx = int(M['m10'] / M['m00'])  # 计算重心
    cy = int(M['m01'] / M['m00'])  # 计算重心
    new_centroid = [cx, cy]
    # cv2.circle(frame, (new_centroid[0], new_centroid[1]), 5, (255,0,255), 5)
    # dist = np.sqrt((old_centroid[0]-new_centroid[0])**2 + (old_centroid[1]-new_centroid[1])**2)
    if (old_centroid[0]-new_centroid[0]) != 0:
        theta = (old_centroid[1]-new_centroid[1]) / (old_centroid[0]-new_centroid[0])
    else:
        theta = 0
    delta_vector = (old_centroid[0]-new_centroid[0], old_centroid[1]-new_centroid[1], theta)
    insert_frame = imutils.translate(frame, new_centroid[0]-old_centroid[0], new_centroid[1]-old_centroid[1])
    insert_frame = imutils.rotate(frame, np.arctan(theta / 4), (new_centroid[0], new_centroid[1]))
    # cv2.imshow("insert", blurred)

    frame = frame[50:300, 50:350]

    if fps._numFrames % 2 == 1:
        # 帧率计数
        end_time = datetime.datetime.now()
        two_frame_time = (end_time - start_time).total_seconds()
        current_fps = (1 / (two_frame_time / 2))
    cv2.putText(frame, str(int(current_fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 出图
    cv2.imshow("Frame", frame)

    old_centroid = [cx, cy]

    # 播放插入帧
    if fps._numFrames % 2 == 1:
        cv2.imshow("Frame", insert_frame[50:350, 50:350])

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # 更新FPS计数
    fps.update()

fps.stop()
camera.stop()
cv2.destroyAllWindows()
