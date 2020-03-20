import os
import random
import cv2
import numpy as np
from Algorithm.OCR.utils import newNet
from Algorithm.utils.Finder import meterFinderBySIFT
from Algorithm.utils.boxRectifier import boxRectifier
from configuration import *
import copy
from Algorithm.OCR.ocr.demo import ocr
import shutil, os

# todo 数字表
class DeleteFiles(object):
    # todo 这个类只是为了 删除文件，这里没有用到，可以不用看
    def __init__(self, pathDir):
        self.pathDir = pathDir
    def delete_files(self):
        os.chdir(self.pathDir)
        fileList = list(os.listdir())
        for file in fileList:
            if os.path.isfile(file):
                os.remove(file)
                # print("delete successfully")
            else:
                shutil.rmtree(file)


import os
CUR_PATH = r'./Algorithm/OCR/ocr/demo_image'   # todo 因为数字表的识别需要用到 ocr，所以这个路径存的是ocr要识别的图片
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def digitPressure(image, allinfo):
    '''

    :param image:   识别的图片
    :param allinfo:   对应的 所有的 json文件
    :return:     返回识别结果
    '''
    del_file(CUR_PATH)  # todo 每次调用函数时，需要把上次存的图片删除，才能存储本次要识别的图片
    # delete_file.delete_files()
    template = None
    flag = None
    info = None
    if type(allinfo) is list:  # todo 用传入的 多个模板 进行一个个的匹配，有一个匹配到则跳出循环

        for i in range(len(allinfo)):
            info = allinfo[i]
            # todo template 是根据模板 提取出图片上的区域
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

    # todo 这时  要识别的区域和对应的json文件已经准备好，开始进行数字识别
    myRes = rgbRecognize(template, info)

    merge = info["merge"]  # todo 上下两行数字是否进行 合并
    decimal = info["decimal"]  # todo 小数点的位置

    # todo 这个只是对识别结果进行一些格式上的处理
    for i in range(len(myRes)):
        temp = ""
        for j, c in enumerate(myRes[i]):
            if c != "?":
                temp += c
            elif j != 0:
                temp += str(random.randint(0, 9))
        myRes[i] = float(temp) if temp != "" else 0.0
    if  merge=='true' :
        a = myRes[0]*10**decimal[1]+myRes[1]
        myRes.clear()
        myRes.append(a)
    return myRes


def rgbRecognize(template, info):
    # todo 由标定点得到液晶区域
    dst = boxRectifier(template, info)  # todo 投影变换
    # 读取标定信息
    height,weight = dst.shape[:2]
    decimal = info["decimal"]
    merge = info["merge"]
    # todo 开始 分割 图片，将分割下来的图片 保存到 ocr 要识别的路径下面，也就是 CUR_PATH 这个路径
    little_height = height // len(decimal)
    for i in range(len(decimal)):
        if i==len(decimal)-1:
            img = dst[i * little_height:height, :]
        else:
            img = dst[i*little_height:(i+1)*little_height,:]
        cv2.imwrite("./Algorithm/OCR/ocr/demo_image/"+str(i)+".jpg",img)  # todo 保存分割下来的图片
    Res = ocr()   # todo 调用ocr 进行识别，ocr() 函数对自动 到 CUR_PATH 路径下读取图片，进行识别
    myRes = []
    # todo 根据小数点位置，是否merge 对识别结果进行处理
    for i,item in enumerate(Res):
        if i==0 and merge=='true':
            numlist = list(item)
            num = "".join(numlist)
            myRes.append(num)
        else:
            numlist = list(item)
            numlist.insert(decimal[i],'.')
            num = "".join(numlist)
            myRes.append(num)
    # 网络初始化



    if ifShow:
        cv2.imshow("rec", dst)
        cv2.imshow("template", template)
        print(myRes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return myRes




