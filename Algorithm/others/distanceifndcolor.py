import cv2
import numpy as np
import json
from Algorithm.others.colordetect import meterFinderNoinfoBySIFT


def FindcolorByDistance(image, info):
    """
    识别含有红色高亮数字区域的模块
    :param image: 输入图片
    :param info:  标记信息
    :return:
    """
    # x = 1
    # y = 1
    # xlen = image.shape[0]
    # ylen = image.shape[1]
    # xbigin = 0
    # ybigin = 0
    result = []
    x = info["xnum"]
    y = info["ynum"]
    xlen = info["xlen"]
    ylen = info["ylen"]
    xbigin = info["xbigin"]
    ybigin = info["ybigin"]
    # print(int(xbigin * ybigin * 0.3 * 255))
    for i in range(x):
        for j in range(y):
            preImage = image[xbigin + xlen * j:xbigin + xlen * (j + 1), ybigin + ylen * i:ybigin + ylen * (i + 1)]
            # cv2.namedWindow("preImage", cv2.WINDOW_NORMAL)
            # cv2.imshow("preImage", preImage)
            # cv2.waitKey(0)
            ImageKey = np.sum(preImage)
            if ImageKey > xbigin * ybigin * 0.1 * 255:
                result.append(1)
            else:
                result.append(0)
    return result


def distanceifndcolor(img, info):
    # 图像预处理
    # imgtemp = cv2.imread("D:/PycharmProjects/nandacode/meterReader/info/20190522/template/30-1_1.jpg")
    imgtemp = info["template"]
    imgsift = meterFinderNoinfoBySIFT(img, imgtemp)  # SIFT匹配
    # print(imgsift.shape)
    gray = cv2.cvtColor(imgsift, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    ret, thresh = cv2.threshold(gray, gray.max() - 25, gray.max(), cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=3)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    # print(imgtemp.shape,thresh.shape)
    # cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)
    result = FindcolorByDistance(thresh, info)
    # print(result)
    return result


if __name__ == '__main__':
    img = cv2.imread("D:/PycharmProjects/nandacode/meterReader/info/20190522/image/30-1.jpg")
    file = open("D:/PycharmProjects/nandacode/meterReader/info/20190522/config/30-1_1.json")
    data = json.load(file)
    distanceifndcolor(img, data)
