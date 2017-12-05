from frame_anticipation import *
import cv2
import imutils


frame_list = load_training_frame("./pic")
next_frame = guess_next_frame(frame_list)
cv2.imshow("Next Frame", imutils.resize(next_frame, width=500))
cv2.waitKey()
