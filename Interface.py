import json
import os
import cv2
from Algorithm.absorb import absorb
from Algorithm.Blenometer import checkBleno
from Algorithm.SF6 import SF6Reader
from Algorithm.oilTempreture import oilTempreture
from Algorithm.videoDigit import videoDigit

from Algorithm.arrest.countArrester import countArrester
from Algorithm.arrest.doubleArrester import doubleArrester

from Algorithm.pressure.digitPressure import digitPressure
from Algorithm.pressure.normalPressure import normalPressure
from Algorithm.pressure.colorPressure import colorPressure
from Algorithm.onoff.Switch import switch
from Algorithm.onoff.onoffIndoor import onoffIndoor
from Algorithm.onoff.onoffOutdoor import onoffOutdoor
from Algorithm.onoff.onoffBatteryScreen import onoffBattery
from Algorithm.onoff.readyStatus import readyStatus
from Algorithm.onoff.springStatus import springStatus
from Algorithm.onoff.contactStatus import contactStatus

from Algorithm.others.colordetect import colordetect
from Algorithm.others.Cabinet_indicator import indicatorimg
from Algorithm.others.Knob_status import knobstatus
from Algorithm.others.distanceifndcolor import distanceifndcolor

from Algorithm.colorIndicator import colorIndicator

from configuration import *


def meterReaderCallBack(image, info):  #todo 开始真正调用相应 的识别函数
    """call back function"""
    if info[0]["type"] == None:
        return "meter type not support!"
    else:
        return info[0]["type"](image, info)


def getInfo(ID):

    """
     # todo
     # todo 通过表的ID，来获取这个表的多个json文件
     #todo 这样做 是为了防止一个模板匹配不到，可以用多个模板进行匹配


    :param ID: 表的 ID
    :return: 一个列表，列表内容是这个表的 多个json文件
    """
    info = []
    config = os.listdir(configPath)
    for i in range(1,200):
        path = ID + "_" + str(i) + ".json"  # 形式为： 1-1_1_1.json,1-1_1_2.json 最后一个数字代表第几个模板
        oneconfigPath = os.path.join(configPath,path)

        if path in config:  # 如果 json文件存在，则加载，否则，跳出循环
            file = open(oneconfigPath)
            info_1 = json.load(file)
            onetemplatePath = os.path.join(templatePath,ID  + "_"+str(i)+".jpg")

            info_1["template"] = cv2.imread(onetemplatePath)  # 加载  模板图片

            if info_1["type"] == "absorb":    # todo type表示表的类型，以及 调用对应函数的名字
                info_1["type"] = absorb
            elif info_1["type"] == "digitPressure":
                info_1["type"] = digitPressure
            elif info_1["type"] == "normalPressure":
                info_1["type"] = normalPressure
            elif info_1["type"] == "switch":
                info_1["type"] = switch
            elif info_1["type"] == "contact":
                info_1["type"] = contactStatus
            elif info_1["type"] == "colorPressure":
                info_1["type"] = colorPressure
            elif info_1["type"] == "SF6":
                info_1["type"] = SF6Reader
            elif info_1["type"] == "countArrester":
                info_1["type"] = countArrester
            elif info_1["type"] == "doubleArrester":
                info_1["type"] = doubleArrester
            elif info_1["type"] == "oilTempreture":
                info_1["type"] = oilTempreture
            elif info_1["type"] == "blenometer":
                info_1["type"] = checkBleno
            elif info_1["type"] == "onoffIndoor":
                info_1["type"] = onoffIndoor
            elif info_1["type"] == "onoffOutdoor":
                info_1["type"] = onoffOutdoor
            elif info_1["type"] == "onoffBattery":
                info_1["type"] = onoffBattery
            elif info_1["type"] == "videoDigit":
                info_1["type"] = videoDigit
            elif info_1["type"] == "ready":
                info_1["type"] = readyStatus
            elif info_1["type"] == "spring":
                info_1["type"] = springStatus
            elif info_1["type"] == "colordetect":
                info_1["type"] = colordetect
            elif info_1["type"] == "cabinetindicator":
                info_1["type"] = indicatorimg
            elif info_1["type"] == "Knob":
                info_1["type"] = knobstatus
            elif info_1["type"] == "colorIndicator":
                info_1["type"] = colorIndicator
            elif info_1["type"] == "distanceifndcolor":
                info_1["type"] = distanceifndcolor
            else:
                info_1["type"] = None

            info.append(info_1)   # todo 将 已经保存的json文件 一个个的添加到 列表里，并返回
        else:
            break
    return info


def meterReader(recognitionData, meterIDs):
    """
    global interface
    :param recognitionData: image or video  图片  或者 图片
    :param meterIDs: list of meter ID
    :return:
    """
    # results = {}
    newresults = []
    # results = -1

    for i, ID in enumerate(meterIDs):

        # get info from file
        info = getInfo(ID)
        if str(info[0]["type"])[10:13] == 'vid':      #todo  如果 识别类型是视频
            newresults = meterReaderCallBack(recognitionData, info)# todo 开始调用识别函数，返回识别结果


            # return results
        else:   # 识别类型是图片
            ROI = recognitionData
            try:
                results = meterReaderCallBack(ROI, info)
                newresults = newresults + results    # todo 开始调用识别函数，返回识别结果
            except AttributeError:
                print("Error in ", ID)
                newresults = [0]

    return newresults

