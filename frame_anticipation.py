import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import regression_tool
from auto_canny import auto_canny
import imutils


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


def translate(image, x, y, border_value=0):
    # define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=border_value)
    # return the translated image
    return shifted


def guess_next_frame(frame_list):
    h = frame_list[0].shape[0]
    w = frame_list[0].shape[1]

    centroid_list_x = []  # 数量等于frame_list
    centroid_list_y = []  # 数量等于frame_list
    dist_list = []  # 数量等于frame_list，最后一个为预测结果
    slope_list = []  # 数量等于frame_list，最后一个为预测结果
    contour_list = []  # 数量等于frame_list

    # 求下一帧的图像重心，并加到centroid_list的最后一项
    for i in range(len(frame_list)):
        tmp = frame_list[i].copy()
        tmp = cv2.GaussianBlur(tmp, (13, 13), 0)
        tmp = auto_canny(tmp)

        contour_list.append(cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2])

        # for_show = np.dstack((frame_list[i].copy(), frame_list[i].copy(), frame_list[i].copy()))
        for_show = frame_list[i].copy()
        contour_list[i] = sorted(contour_list[i], key=cv2.contourArea, reverse=True)
        cv2.drawContours(for_show, contour_list[i], 0, (128, 128, 255), -1)
        # cv2.imshow("Frame", for_show)
        # cv2.waitKey()

        M = cv2.moments(contour_list[i][0])  # 计算矩
        cx = int(M['m10'] / M['m00'])  # 计算重心
        cy = int(M['m01'] / M['m00'])  # 计算重心
        centroid_list_x.append(cx)
        centroid_list_y.append(cy)

        if i > 0:
            dist_list.append(np.sqrt((centroid_list_x[i] - centroid_list_x[i - 1]) ** 2 + (
                    centroid_list_y[i] - centroid_list_y[i - 1]) ** 2))

            if (centroid_list_x[i] - centroid_list_x[i - 1]) != 0:
                slope_list.append(
                    (centroid_list_y[i] - centroid_list_y[i - 1]) / (centroid_list_x[i] - centroid_list_x[i - 1]))
            else:
                slope_list.append(np.inf)

    next_theta_param = regression_tool.expfit(np.arange(len(slope_list)), np.round(slope_list, 9))[0]
    next_theta = regression_tool.expfunc(len(slope_list), next_theta_param[0], next_theta_param[1], next_theta_param[2])

    next_dist = np.average(dist_list)

    if centroid_list_x[-2] < centroid_list_x[-1]:
        next_centroid_x = next_dist / np.sqrt(next_theta ** 2 + 1) + centroid_list_x[-1]
    else:
        next_centroid_x = - next_dist / np.sqrt(next_theta ** 2 + 1) + centroid_list_x[-1]

    if centroid_list_y[-2] < centroid_list_y[-1]:
        next_centroid_y = next_dist * next_theta / np.sqrt(next_theta ** 2 + 1) + centroid_list_y[-1]
    else:
        next_centroid_y = - next_dist * next_theta / np.sqrt(next_theta ** 2 + 1) + centroid_list_y[-1]

    centroid_list_x.append(next_centroid_x)
    centroid_list_y.append(next_centroid_y)

    # plt.plot(centroid_list_x, centroid_list_y, "-or")
    # plt.show()

    a = next_centroid_x - centroid_list_x[-2]
    b = next_centroid_y - centroid_list_y[-2]

    next_frame = translate(frame_list[-1], next_centroid_x - centroid_list_x[-2],
                                   next_centroid_y - centroid_list_y[-2], border_value=255)
    # cv2.imshow("1", next_frame)
    next_frame = cv2.bitwise_and(next_frame, frame_list[-1])
    # cv2.imshow("Next Frame", next_frame)

    return next_frame
