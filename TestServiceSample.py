import requests
import base64
import json
import os
import time
import cv2
import multiprocessing
from configuration import *

from Interface import meterReader


def startServer():
    os.system("python FlaskService.py")


def startClient(results):
    images = os.listdir("info/20190423/IMAGES/Pic_2")
    for im in images:
        path = "info/20190410/IMAGES/Pic_2/" + im
        data = json.dumps({
            "path": path,
            "imageID": im.split('.')[0] + "_1"
        })
        print(path, im)
        print(data)
        r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
        print(r.text)
        receive = json.loads(r.text)
        print(im, receive)

        results.append(True)

def codecov(imgPath):
    images = os.listdir(imgPath)
    config = os.listdir(configPath)
    config = [s.split(".")[0][:-2] for s in config]
    newconfig = []
    for item in config:
        if item not in newconfig:
            newconfig.append(item)
    for im in images:
        image = cv2.imread(imgPath + "/" + im)  # todo 读取图片
        print(im)
        pos = im.split(".")[0].split("-")
        # cfg = im.split(".")[0]+"_1"
        for i in range(1, 6):
            cfg = pos[0] + "-" + pos[1] + "_" + str(i)
            if cfg  in newconfig:
                receive2 = meterReader(image, [cfg])   # todo 调用meterReader 接口
                print(cfg, receive2)
        # cv2.imshow("ds",image)
        # cv2.waitKey()
    print("codecov done")
def getLabelPictures():
    """
    截取视频中每隔固定时间出现的帧
    :param videoCapture: 输入视频
    :return: 截取的帧片段
    """
    video_path = "info/20190609/images/video"
    for file in os.listdir(video_path):
        video = cv2.VideoCapture(os.path.join(video_path, file))
        pictures = []
        cnt = 0
        skipFrameNum = 50
        while True:
            ret, frame = video.read()
            # print(cnt, np.shape(frame))
            cnt += 1
            if(cnt==skipFrameNum):
                cv2.imwrite(video_path+file[:-4]+'.jpg',frame)
                break
        video.release()
    return pictures
def testVideo():
    #video_path = "info/20190128/IMAGES/video_"
    video_path = "info/20191207/image/video"
    for file in os.listdir(video_path):
        if file.startswith(".DS"):
            continue
        video = cv2.VideoCapture(os.path.join(video_path, file))  #todo 读取视频
        start = time.clock()
        result = meterReader(video, [file[:-5] + "1_1"])   # todo 调用meterReader 接口
        end = time.clock()
        print(file, result)
        print(end-start)
    print("codecov done")


if __name__ == "__main__":
    # Service Test

    # serverProcess = multiprocessing.Process(target=startServer)
    # results = multiprocessing.Manager().list()
    # clientProcess = multiprocessing.Process(target=startClient, args=(results,))
    # serverProcess.start()
    # time.sleep(30)
    # clientProcess.start()
    # clientProcess.join()
    # serverProcess.terminate()

    # Single Test

    # testReadyStatus()
    # codecov("info/20190128/IMAGES/image")
    # codecov("info/20190128/IMAGES/Pic_0225")
    # codecov("info/20190128/IMAGES/Pic_0226")
    # codecov("info/20190128/IMAGES/video_")

    # codecov("info/20190410/IMAGES/Pic")
    # codecov("info/20190410/IMAGES/Pic_2")
    # codecov("info/20191206/image")
    codecov("info/20191114/image/")
    #
    # codecov("info/20190416/IMAGES/image")
    #getLabelPictures()
    # testVideo()

