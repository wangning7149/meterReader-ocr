import os
import random

import cv2
import numpy as np
from Algorithm.utils.Finder import meterFinderBySIFT
from configuration import *
from Algorithm.OCR.onoff.model1 import switchNet
import copy
import os
import torch


def switch(image, allinfo):

    # delete_file.delete_files()
    template = None
    flag = None
    info = None
    if type(allinfo) is list:

        for i in range(len(allinfo)):
            info = allinfo[i]

            template,flag= meterFinderBySIFT(image, info)   #todo flag==0 则是没有匹配到，继续用下一个模板匹配

            if flag==0:
                continue
            else:
                break
    else:
        template, flag = meterFinderBySIFT(image, allinfo)
        info = copy.deepcopy(allinfo)
    if flag == 0:
        # print('not find template!!!')
        pass
    h,w,_ = template.shape
    left = template[:,0:w//2].copy()

    # todo   存储数据

    #store_num = len(os.listdir("./store"))
    #cv2.imwrite("./store/left_"+str(store_num)+".jpg",left)


    left = cv2.resize(left,(40,40))
    left = np.expand_dims(left,0)   # todo 分为左右两个部分
    right = template[:,w//2:].copy()

    # # todo   存储数据
    # store_num = len(os.listdir("./store"))
    # cv2.imwrite("./store/right_"+str(store_num)+".jpg",right)

    right = cv2.resize(right,(40,40))
    right = np.expand_dims(right,0)
    net = switchNet()  # todo 加载模型
    net.load_state_dict(torch.load("Algorithm/OCR/onoff/model/left_switch_net.pth"))
    input = torch.tensor(left,dtype=torch.float32).permute((0,3,1,2))

    out = net(input)
    out = out.detach().numpy()    # 开 是 1    关 是 0
    left_index = np.argmax(out)
    net1 = switchNet()   # todo 加载模型
    net1.load_state_dict(torch.load("Algorithm/OCR/onoff/model/right_switch_net.pth"))
    input = torch.tensor(right, dtype=torch.float32).permute((0, 3, 1, 2))

    out = net(input)
    out = out.detach().numpy()  # 开 是 1    关 是 0
    right_index = np.argmax(out)
    return [left_index,right_index]








