import cv2
import glob
import numpy as np
import regression_tool


def get_frame_from_queue(queue, quantity):
    frame_list = []
    for i in range(quantity):
        frame_list.append(queue.get())

    return frame_list


def play_frame_from_queue(queue, play_on):
    cv2.imshow(play_on, queue.get())


def load_training_frame(path):
    frame_list = []
    for frame in glob.glob(path + "/*.jpg"):
        frame_list.append(cv2.imread(frame[frame.rfind("/") + 1:], cv2.IMREAD_GRAYSCALE))

    return frame_list


def guess_next_frame(frame_list):
    centroid_list = []  # 数量等于frame_list
    dist_list = []  # 数量等于frame_list，最后一个为预测结果
    theta_list = []  # 数量等于frame_list，最后一个为预测结果

    for i in range(len(frame_list)):
        M = cv2.moments(frame_list[i])  # 计算矩
        cx = int(M['m10'] / M['m00'])  # 计算重心
        cy = int(M['m01'] / M['m00'])  # 计算重心
        centroid_list.append([cx, cy])

        if i > 0:
            dist_list.append(np.sqrt((centroid_list[i][0]-centroid_list[i-1][0])**2 + (centroid_list[i][1]-centroid_list[i-1][1])**2))

            if (centroid_list[i][0]-centroid_list[i-1][0]) != 0:
                theta_list.append((centroid_list[i][1]-centroid_list[i-1][1]) / (centroid_list[i][0]-centroid_list[i-1][0]))
            else:
                theta_list.append(np.inf)

    next_frame = frame_list[-1]

    next_theta = regression_tool.polyfit(np.arange(len(frame_list)))

    return next_frame
