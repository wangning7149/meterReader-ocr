import cv2
import numpy as np

from Algorithm.utils.Finder import meterFinderByTemplate, meterFinderBySIFT

import copy
def colorIndicator(ROI, allinfo):
    res = [0]
    template = None
    flag = None
    info = None
    if type(allinfo) is list:

        for i in range(len(allinfo)):
            info = allinfo[i]

            template, flag = meterFinderBySIFT(ROI, info)  # todo flag==0 则是没有匹配到，继续用下一个模板匹配
            if flag == 0:
                continue
            else:
                break

    else:
        template, flag = meterFinderBySIFT(ROI, allinfo)
        info = copy.deepcopy(allinfo)
    if flag == 0:
        print('not find template!!!')
    HSV = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    # color = [np.array([26, 43, 46]), np.array([34, 255, 255])]
    # todo 根据颜色进行识别，结合实际图片
    color = [np.array([11, 43, 46]), np.array([34, 255, 255])]
    Lower = color[0]
    Upper = color[1]
    mask = cv2.inRange(HSV, Lower, Upper)
    upmask = mask[int(0.25*mask.shape[0]):int(0.5*mask.shape[0]), :]
    upmask = cv2.bitwise_and(np.ones(upmask.shape, np.uint8), upmask)
    if np.sum(upmask) / upmask.shape[0]*upmask.shape[1] > 0.2:
        res = [1]
    return res
