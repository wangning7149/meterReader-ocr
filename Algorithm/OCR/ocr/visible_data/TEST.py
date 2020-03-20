import cv2
import base64
import numpy as np
def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code
#
#
def base64_to_image(base64_code):
    # base64解码
    # img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(base64_code, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img
# with open('2.png','rb') as f:
#     a = f.read()
# # image_np = cv2.imread('2.png')
# # image_code = image_to_base64(image_np)
# img = base64_to_image(a)
# img = cv2.resize(img,(0,0),fx=4,fy=4)
# cv2.imshow("dsa",img)
# cv2.waitKey(0)
""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511621)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)

    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split(' ')

        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':

    createDataset('./image','gtfiles.txt','out')
    # env = lmdb.open("./out")

    # 参数write设置为True才可以写入
    # txn = env.begin(write=True)
    # ############################################添加、修改、删除数据
    #
    # # 添加数据和键值
    # txn.put(key='1', value='aaa')
    # txn.put(key='2', value='bbb')
    # txn.put(key='3', value='ccc')
    #
    # # 通过键值删除数据
    # txn.delete(key='1')
    #
    # # 修改数据
    # txn.put(key='3', value='ddd')
    #
    # # 通过commit()函数提交更改
    # txn.commit()
    ############################################查询lmdb数据
    # txn = env.begin()

    # get函数通过键值查询数据
    #

    # 通过cursor()遍历所有数据和键值
    # for key, value in txn.cursor():
    #     img =base64_to_image(value)
    #     print(img.shape)
    #     cv2.imshow("das",img)
    #     cv2.waitKey()




