import cv2
import numpy as np

from configuration import *
from Algorithm.utils.AngleFactory import AngleFactory


def getPoints(center, radious, angle):
    # todo 计算在 固定的角度下  将所有包含的点存起来
    res = []
    farthestPointX = int(center[0] + radious * np.cos(angle / 180 * np.pi))
    farthestPointY = int(center[1] + radious * np.sin(angle / 180 * np.pi))

    for ro in range(radious // 3, radious):
        for theta in range(angle - 2, angle + 2):  # todo 在当前角度取一个范围
            angleTemp = theta / 180 * np.pi       #  todo 存储 这个范围里面的所有点的坐标
            x, y = int(ro * np.cos(angleTemp)) + center[0], int(ro * np.sin(angleTemp)) + center[1]
            res.append([x, y])
    return res, [farthestPointX, farthestPointY]


def EuclideanDistance(pt1, pt2):  # todo 计算两点之间的距离
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def scanPointer(meter, info):
    '''

    :param meter:  指针表的区域
    :param info:   json文件
    :return:  指针的读数
    '''
    # todo
    center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])  # todo 指针表的中心
    start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])     # todo 量程的起点坐标
    end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])         # todo 量程的终点 坐标
    startVal = int(info["startValue"])    #  todo 量程的 最小值
    endVal = int(info["totalValue"])     # todo 量程的 最大值
    if meter.shape[0] > 500:  # todo 只是对区域大小有一个限制，无关紧要
        fixHeight = 300
        fixWidth = int(meter.shape[1] / meter.shape[0] * fixHeight)
        resizeCoffX = fixWidth / meter.shape[1]
        meter = cv2.resize(meter, (fixWidth, fixHeight))

        start = (start * resizeCoffX).astype(np.int32)
        end = (end * resizeCoffX).astype(np.int32)
        center = (center * resizeCoffX).astype(np.int32)
    # todo 根据中心点的起点  计算半径  ，
    # todo 作用：以仪表中心点为圆心，中心点到量程起点的距离为半径，开始从头到尾进行扫描
    radious = int(EuclideanDistance(start, center))

    # todo 对图片进行 预处理，转化为二值图,
    #  todo 以下一系列的操作，就是为了得到 黑色背景，白色前景的二值图
    src = cv2.GaussianBlur(meter, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 11)  #todo
    mask = np.zeros((src.shape[0], src.shape[1]), np.uint8)
    cv2.circle(mask, (center[0], center[1]), radious, (255, 255, 255), -1)
    thresh = cv2.bitwise_and(thresh, mask)
    cv2.circle(thresh, (center[0], center[1]), int(radious / 3), (0, 0, 0), -1)
    thresh = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # todo 计算 扫描时 最小角度  和  对大角度
    startAngle = int(
        AngleFactory.calAngleClockwise(startPoint=np.array([center[0] + 100, center[1]]), centerPoint=center,
                                       endPoint=start) * 180 / np.pi)
    endAngle = int(AngleFactory.calAngleClockwise(startPoint=np.array([center[0] + 100, center[1]]), centerPoint=center,
                                                  endPoint=end) * 180 / np.pi)
    # print(startAngle, endAngle)
    if endAngle <= startAngle:
        endAngle += 360
    maxSum = 0  # todo 下面的for循环中，计算每个角度下与指针重合的点的个数（thisSum），maxSum是他们的最大值
    outerPoint = start
    for angle in range(startAngle - 10, endAngle + 10):
        pts, outPt = getPoints(center, radious, angle)  # todo 计算并保存当前这个角度下的所有点
        thisSum = 0  # todo 统计在当前角度下，与指针重合的点的个数
        showImg = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

        for pt in pts: # todo pts里面的存的是当前角度下所有点的坐标，
            cv2.circle(showImg, (pt[0], pt[1]), 2, (0, 0, 255), -1)
            # todo 这时候是指针真值对比
            if thresh[pt[1], pt[0]] != 0:
                # todo 因为二值图是黑色背景，白色前景，所以有指针的位置的像素值是255，像素值是0的位置就不是指针
                thisSum += 1

        # cv2.circle(showImg, (outPt[0], outPt[1]), 2, (255, 0, 0), -1)
        # cv2.imshow("img", showImg)
        # cv2.waitKey(1)
        if thisSum > maxSum:    # todo 选一个 重合点数最高的
            maxSum = thisSum
            outerPoint = outPt  # todo 这时的outerPoint就是指针顶点的坐标

    if start[0] == outerPoint[0] and start[1] == outerPoint[1]:# todo 这种情况是指针在量程开始的位置
        degree = startVal
    elif end[0] == outerPoint[0] and end[1] == outerPoint[1]:   # todo 这种情况是指针在量程最大的位置
        degree = endVal
    else:# todo 有了json文件，这种情况不存在
        if start[0] == end[0] and start[1] == end[1]:
            end[0] -= 1
            end[1] -= 3
        # todo  这时就可以根据量程大小 计算 读数了
        degree = AngleFactory.calPointerValueByOuterPoint(start, end, center, outerPoint, startVal, endVal)

    # small value to zero
    if degree - startVal < 0.05 * (endVal - startVal):
        degree = startVal

    if ifShow:  # todo 可视化的时候用
        print("degree {:.2f} startPoint {}, endPoint{}, outPoint {}".format(degree, start, center, outerPoint))
        cv2.circle(meter, (outerPoint[0], outerPoint[1]), 10, (0, 0, 255), -1)
        cv2.line(meter, (center[0], center[1]), (outerPoint[0], outerPoint[1]), (0, 0, 255), 5)
        cv2.line(meter, (center[0], center[1]), (start[0], start[1]), (255, 0, 0), 3)
        cv2.line(meter, (center[0], center[1]), (end[0], end[1]), (255, 0, 0), 3)

        thresh = np.expand_dims(thresh, axis=2)
        thresh = np.concatenate((thresh, thresh, thresh), 2)
        meter = np.hstack((meter, thresh))

        cv2.imshow("test", meter)
        cv2.waitKey(0)
    return int(degree*100)/100
