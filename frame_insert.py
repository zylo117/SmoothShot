import imutils
from imutils.video.videostream import VideoStream
import cv2
from imutils.video import FPS
import datetime

# if a video path was not supplied, grab the reference to the webcam
camera = VideoStream(src=0).start()

fps = FPS().start()
current_fps = 0

# keep looping
while True:
    frame = camera.read()

    if fps._numFrames % 2 == 0:
        # 帧率计数
        start_time = datetime.datetime.now()

    frame = imutils.resize(frame, width=400)

    # 生成插入帧
    if fps._numFrames % 2 == 0:
        insert_frame = frame

    if fps._numFrames % 2 == 1:
        cv2.merge([insert_frame, frame])

    if fps._numFrames % 2 == 1:
        # 帧率计数
        end_time = datetime.datetime.now()
        two_frame_time = (end_time - start_time).total_seconds()
        current_fps = (1 / (two_frame_time / 2))
    cv2.putText(frame, str(int(current_fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 出图
    cv2.imshow("Frame", frame)

    # 播放插入帧
    if fps._numFrames % 2 == 1:
        cv2.imshow("Frame", insert_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # 更新FPS计数
    fps.update()

fps.stop()
camera.stop()
cv2.destroyAllWindows()
