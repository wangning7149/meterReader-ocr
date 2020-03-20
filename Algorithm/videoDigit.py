import cv2
import sys
import os
import json
import shutil
import numpy as np
import torch
from collections import defaultdict, Counter
import copy
from Algorithm.OCR.ocr.demo import ocr
from Algorithm.pressure.digitPressure import digitPressure
from Algorithm.utils.Finder import meterFinderBySIFT
# from Algorithm.utils.Finder import meterFinderBySIFT, meterReginAndLocationBySIFT
from Algorithm.OCR.character.characterNet import characterNet,rgbCharacterNet
from Algorithm.utils.boxRectifier import boxRectifier
global temp
def videoDigit(video, allinfo):
    """
    :param video: VideoCapture Input
    :param info: info
    :return:
    """
    net = characterNet(is_rgb=True)  # todo 创建并加载 模型
    net.load_state_dict(torch.load("Algorithm/OCR/character/rbg2.pkl",map_location='cpu'))

    pictures = getPictures(video)  # todo 获得视频的帧，有少量重复帧

    def emptyLsit():
        return []

    imagesDict = defaultdict(emptyLsit)
    imagesDict2 = {'A':-1,'B':-1,'C':-1}  # todo 防止没有识别出 ABC 三种帧  根据视频观察出 每隔120帧会变换利用这个特点进行第二种方法的编写
    box = None
    newinfo = None
    newchange = None
    saveframe = []

    for i, frame in enumerate(pictures):
       # res = digitPressure(frame, copy.deepcopy(info))
        for j in range(len(allinfo)):

            eachinfo = copy.deepcopy(allinfo[j])
            template,flag= meterFinderBySIFT(frame, eachinfo)   # todo flag==0 则是没有匹配到，继续用下一个模板匹配
            if flag == 0:
                continue
            else:
                break
        if flag == 0:
            # print("not find template!!!")
            pass
        info = copy.deepcopy(eachinfo)
        index, charimg = checkFrame(i, net, template, eachinfo)  # todo 识别当前的帧到底是属于哪一个类别

        if index < 3:
            res = rgbRecognize(template, eachinfo)  # todo 得出 数字的识别结果，存到对应的key值下
            imagesDict[chr(index+ord('A'))] += [res]
            if len(imagesDict['A']) != 0 and imagesDict2['B'] == -1:
                imagesDict2['B'] = (i + 3) % len(pictures)
            if len(imagesDict['B']) != 0 and imagesDict2['C'] == -1:
                imagesDict2['C'] = (i + 3) % len(pictures)
            if len(imagesDict['C']) != 0 and imagesDict2['A'] == -1:
                imagesDict2['A'] = (i + 6) % len(pictures)
    for key,val in imagesDict.items():
        if len(val) == 0:
            res = digitPressure(pictures[imagesDict2[key]], allinfo)
            imagesDict[key] += [res]


    imagesDict = getResult(imagesDict)   #todo  对 结果做一些处理

    return imagesDict

def getResult(dicts):
    global temp
    newdicts = {'A': [], 'B': [], 'C': []}
    for ctype, res in dicts.items():
        # print("res ",ctype, res)
        total=[]
        tmp=[]
        for x in res:
            for c in x:
                str1=str(c)
                while len(str1.split('.')[0])<3:
                    str1='0'+str1
                while len(str1.split('.')[1])<2:
                    str1=str1+'0'
                tmp.append(str1)
            total.append(tmp)
            tmp=[]

        firsts = [[c for c in x[0]] for x in total]
        seconds = [[c for c in x[1]] for x in total]

        if len(firsts[0]) > 0 and len(seconds[0]) > 0:
            number = ""
            for j in range(6):
                words = [a[j] for a in firsts]
                #print(words)
                num = Counter(words).most_common(1)
                number = number + num[0][0]
            newdicts[ctype].append(number)
            number = ""
            for j in range(6):
                words = [a[j] for a in seconds]
                num = Counter(words).most_common(1)
                number = number + num[0][0]
            newdicts[ctype].append(number)

    for k,v in newdicts.items():
        if float(v[1]) > 100.1 and float(v[1]) < 120:
            temp = v[1]
            break
    for k,v in newdicts.items():
        if float(v[1]) < 100.1 or float(v[1]) > 120:
            v[1] = temp
    return newdicts

def getPictures(videoCapture):
    """
    截取视频中每隔固定时间出现的帧
    :param videoCapture: 输入视频
    :return: 截取的帧片段
    """
    pictures = []
    cnt = 0
    skipFrameNum = 30
    while True:
        ret, frame = videoCapture.read()
        # print(cnt, np.shape(frame))
        cnt += 1
        if frame is None:
            break
        if cnt % skipFrameNum:
            continue
        pictures.append(frame)

    videoCapture.release()
    return pictures

def checkFrame(count, net, template, info,is_rgb=True):  # todo 识别当前的帧到底是属于哪一个类别
    """
    判断图片的类型A，B，C等
    :param image:  image
    :param info: info
    :return: 出现次序0.1.2.3
    """
    start = [info["startPoint"]["x"], info["startPoint"]["y"]]
    end = [info["endPoint"]["x"], info["endPoint"]["y"]]
    center = [info["centerPoint"]["x"], info["centerPoint"]["y"]]
    charastart = [info["charastartPoint"]["x"], info["charastartPoint"]["y"]]
    charaend = [info["charaendPoint"]["x"], info["charaendPoint"]["y"]]
    characenter= [info["characenterPoint"]["x"], info["characenterPoint"]["y"]]


    imgType = boxRect(template,info,charastart,charaend,characenter)

    imgType = cv2.resize(imgType,(30,36),interpolation=cv2.INTER_CUBIC)
    #imgType = cv2.resize(imgType,(28,36),interpolation=cv2.INTER_CUBIC)
    orimage = imgType.copy()

    imgType = torch.tensor(np.array(imgType, dtype=np.float32))
    imgType=torch.Tensor(imgType).view(3, 30, 36)
    #imgType=torch.Tensor(imgType).view(3, 28, 36)
    imgType = torch.unsqueeze(imgType, 0)

    type_probe = net.forward(imgType)
    type_probe = type_probe.detach().numpy()
    maxIndex = np.argmax(type_probe)

    return maxIndex, orimage


def EuclideanDistance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def rgbRecognize(template, info):  # todo 和指针表的一样
    # 由标定点得到液晶区域
    dst = boxRectifier(template, info)
    # 读取标定信息
    height,weight = dst.shape[:2]
    decimal = info["decimal"]
    merge = info["merge"]
    little_height = height // len(decimal)
    for i in range(len(decimal)):
        if i==len(decimal)-1:
            img = dst[i * little_height:height, :]
        else:
            img = dst[i*little_height:(i+1)*little_height,:]
        cv2.imwrite("./Algorithm/OCR/ocr/demo_image/"+str(i)+".jpg",img)
    Res = ocr()
    myRes = []
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

    merge = info["merge"]
    decimal = info["decimal"]

    for i in range(len(myRes)):
        temp = ""
        for j, c in enumerate(myRes[i]):
            if c != "?":
                temp += c
            elif j != 0:
                temp += str(random.randint(0, 9))
        myRes[i] = float(temp) if temp != "" else 0.0
    if merge == 'true':
        a = myRes[0] * 10 ** decimal[1] + myRes[1]
        myRes.clear()
        myRes.append(a)

    return myRes




def boxRect(templateImage, info,start,end,center):  #todo 投影变换

    if "rectangle" in info:
        width = info["rectangle"]["width"]
        height = info["rectangle"]["height"]
    else:
        width = int(EuclideanDistance(start, center))
        height = int(EuclideanDistance(center, end))

    # todo 计算数字表的矩形外框，并且拉直矫正
    fourth = (start[0] + end[0] - center[0], start[1] + end[1] - center[1])
    pts1 = np.float32([start, center, end, fourth])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(templateImage, M, (width, height))
    return dst