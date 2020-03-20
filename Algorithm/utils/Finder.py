import cv2
import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_distances


def meterFinderByTemplate(image, template):  # todo 没有用到，可以不用看
    """
    locate meter's bbox
    :param image: image
    :param template: template
    :return: bbox image
    """
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    w, h, _ = template.shape

    # for test
    # cv2.imshow("test", img)
    # img = (img * 0.5).astype(np.uint8) # test
    # cv2.imshow("test2", img)
    # cv2.waitKey(0)

    i = 5
    res = cv2.matchTemplate(image, template, methods[i])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    if methods[i] in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        topLeft = minLoc
    else:
        topLeft = maxLoc
    bottomRight = (topLeft[0] + h, topLeft[1] + w)

    return image[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]


def meterLocationFinderBySIFT(image, template):  # todo 项目里没用到
    """
    locate meter's bbox
    :param image: image
    :param template: template
    :return: bbox, left top & right bottom
    """
    templateBlurred = cv2.GaussianBlur(template, (3, 3), 0)
    imageBlurred = cv2.GaussianBlur(image, (3, 3), 0)

    sift = cv2.xfeatures2d.SIFT_create()

    # shape of descriptor n * 128, n is the num of key points.
    # a row of descriptor is the feature of related key point.
    templateKeyPoint, templateDescriptor = sift.detectAndCompute(templateBlurred, None)
    imageKeyPoint, imageDescriptor = sift.detectAndCompute(imageBlurred, None)

    # for debug
    # templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
    # imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
    # cv2.imshow("template", templateBlurred)
    # cv2.imshow("image", imageBlurred)

    # match
    bf = cv2.BFMatcher()
    # k = 2, so each match has 2 points. 2 points are sorted by distance.
    matches = bf.knnMatch(templateDescriptor, imageDescriptor, k=2)

    # The first one is better than the second one
    good = [[m] for m, n in matches if m.distance < 0.7 * n.distance]

    # distance matrix
    templatePointMatrix = np.array([list(templateKeyPoint[p[0].queryIdx].pt) for p in good])
    imagePointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good])
    templatePointDistanceMatrix = pairwise_distances(templatePointMatrix, metric="euclidean")
    imagePointDistanceMatrix = pairwise_distances(imagePointMatrix, metric="euclidean")

    # del bad match
    distances = []
    maxAbnormalNum = 15
    for i in range(len(good)):
        diff = abs(templatePointDistanceMatrix[i] - imagePointDistanceMatrix[i])
        # distance between distance features
        diff.sort()
        distances.append(np.sqrt(np.sum(np.square(diff[:-maxAbnormalNum]))))

    averageDistance = np.average(distances)
    good2 = [good[i] for i in range(len(good)) if distances[i] < 2 * averageDistance]

    # for debug
    # matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
    # cv2.imshow("matchImage", matchImage)
    # cv2.waitKey(0)

    matchPointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good2])

    # for p1, p2 in matchPointMatrix:
    #     cv2.circle(image, (int(p1), int(p2)), 0, (255, 0, 0), thickness=50)
    #     print(p1, p2)
    # cv2.imshow("matchImage", image)

    minX = int(np.min(matchPointMatrix[:, 0]))
    maxX = int(np.max(matchPointMatrix[:, 0]))
    minY = int(np.min(matchPointMatrix[:, 1]))
    maxY = int(np.max(matchPointMatrix[:, 1]))
    return minX, minY, maxX, maxY


def meterFinderBySIFT(orign_img, info):
    """
    locate meter's bbox
    :param orign_img: image
    :param info: info
    :return: bbox image
    """
    if str(info["type"])[10:13] != 'vid' or str(info["type"])[10:13] != 'dig':
        x = info["ROI"]["x"]
        y = info["ROI"]["y"]
        w = info["ROI"]["w"]
        h = info["ROI"]["h"]
        # call back
        # cv2.rectangle(recognitionData, (x, y), (x+w, y + h), (255, 0, 0), 3)
        # cv2.imshow("testInput", recognitionData)
        # cv2.waitKey(0)
        if x != 0 or y != 0 or w != 0 or h != 0:
            image = orign_img[y:y + h, x:x + w]
        else:
            image = orign_img
    template = info["template"]


    # cv2.imshow("template", template)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # todo 根据标定信息 获取所有点的信息
    startPoint = (info["startPoint"]["x"], info["startPoint"]["y"])
    centerPoint = (info["centerPoint"]["x"], info["centerPoint"]["y"])
    endPoint = (info["endPoint"]["x"], info["endPoint"]["y"])
    charastartPoint = (info["charastartPoint"]["x"], info["charastartPoint"]["y"])
    charaendPoint = (info["charaendPoint"]["x"], info["charaendPoint"]["y"])
    characenterPoint = (info["characenterPoint"]["x"], info["characenterPoint"]["y"])

    templateBlurred = cv2.GaussianBlur(template, (3, 3), 0)
    imageBlurred = cv2.GaussianBlur(image, (3, 3), 0)

    sift = cv2.xfeatures2d.SIFT_create()
    # todo 分别对 原图 和 模板 计算sift特征点
    templateKeyPoint, templateDescriptor = sift.detectAndCompute(templateBlurred, None)
    imageKeyPoint, imageDescriptor = sift.detectAndCompute(imageBlurred, None)

    # for debug
    # templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
    # imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
    # # cv2.imshow("template", templateBlurred)
    # cv2.imshow("image", imageBlurred)
    # cv2.waitKey(0)

    # match
    bf = cv2.BFMatcher()
    # k = 2, so each match has 2 points. 2 points are sorted by distance.
    # todo  对特征点 进行匹配度的计算
    matches = bf.knnMatch(templateDescriptor, imageDescriptor, k=2)

    # The first one is better than the second one
    # todo 筛选 好的特征点
    good = [[m] for m, n in matches if m.distance < 0.8 * n.distance]
    # todo 计算距离矩阵，根据这个矩阵 会再次筛选
    templatePointMatrix = np.array([list(templateKeyPoint[p[0].queryIdx].pt) for p in good])
    imagePointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good])
    templatePointDistanceMatrix = pairwise_distances(templatePointMatrix, metric="euclidean")
    imagePointDistanceMatrix = pairwise_distances(imagePointMatrix, metric="euclidean")

    # del bad match

    distances = []
    maxAbnormalNum = 15
    for i in range(len(good)):
        diff = abs(templatePointDistanceMatrix[i] - imagePointDistanceMatrix[i])
        # distance between distance features
        diff.sort()
        distances.append(np.sqrt(np.sum(np.square(diff[:-maxAbnormalNum]))))

    averageDistance = np.average(distances)
    # todo 第二次筛选 特征点
    good2 = [good[i] for i in range(len(good)) if distances[i] < 2 * averageDistance]

    # for debug
    # matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
    # cv2.imshow("matchImage", matchImage)
    # cv2.waitKey(0)

    # not match
    if len(good2) < 3:
        # print("not found!")
        return template,0

    # todo 根据选取的特征点，寻找原图与模板之间的 转换矩阵 M
    src_pts = np.float32([templateKeyPoint[m[0].queryIdx].pt for m in good2]).reshape(-1, 1, 2)
    dst_pts = np.float32([imageKeyPoint[m[0].trainIdx].pt for m in good2]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # todo M就是转换矩阵
    matchesMask = mask.ravel().tolist()
    h, w, _ = template.shape

    # todo 将原来json里的点的坐标 根据转换矩阵 全部转换成在匹配到的区域上的坐标
    pts = np.float32(
        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0], [startPoint[0], startPoint[1]], [endPoint[0], endPoint[1]],
         [centerPoint[0], centerPoint[1]],
         [charastartPoint[0], charastartPoint[1]],
         [charaendPoint[0], charaendPoint[1]],
         [characenterPoint[0], characenterPoint[1]]
         ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)  # todo 转换

    # todo 由于匹配到的区域可能并不是水平的，需要进行旋转
    angle = 0.0
    vector = (dst[3][0][0] - dst[0][0][0], dst[3][0][1] - dst[0][0][1])
    cos = (vector[0] * (200.0)) / (200.0 * math.sqrt(vector[0] ** 2 + vector[1] ** 2))
    if (vector[1] > 0):
        angle = math.acos(cos) * 180.0 / math.pi
    else:
        angle = -math.acos(cos) * 180.0 / math.pi
    # print(angle)
    # todo 获得旋转矩阵
    change = cv2.getRotationMatrix2D((dst[0][0][0], dst[0][0][1]), angle, 1)
    # todo 旋转
    src_correct = cv2.warpAffine(image, change, (image.shape[1], image.shape[0]))
    array = np.array([[0, 0, 1]])
    newchange = np.vstack((change, array))
    # todo 将所有点的坐标也需要进行旋转
    newpoints = []
    for i in range(len(pts)):
        point = newchange.dot(np.array([dst[i][0][0], dst[i][0][1], 1]))
        point = list(point)
        point.pop()
        newpoints.append(point)
    # todo 提取模板匹配到的区域
    src_correct = src_correct[int(round(newpoints[0][1])):int(round(newpoints[1][1])),
                  int(round(newpoints[0][0])):int(round(newpoints[3][0]))]

    width = src_correct.shape[1]
    height = src_correct.shape[0]
    if width == 0 or height == 0:
        return template,0

    startPoint = (int(round(newpoints[4][0]) - newpoints[0][0]), int(round(newpoints[4][1]) - newpoints[0][1]))
    endPoint = (int(round(newpoints[5][0]) - newpoints[0][0]), int(round(newpoints[5][1]) - newpoints[0][1]))
    centerPoint = (int(round(newpoints[6][0]) - newpoints[0][0]), int(round(newpoints[6][1]) - newpoints[0][1]))
    charastartPoint = (int(round(newpoints[7][0]) - newpoints[0][0]), int(round(newpoints[7][1]) - newpoints[0][1]))
    charaendPoint = (int(round(newpoints[8][0]) - newpoints[0][0]), int(round(newpoints[8][1]) - newpoints[0][1]))
    characenterPoint = (int(round(newpoints[9][0]) - newpoints[0][0]), int(round(newpoints[9][1]) - newpoints[0][1]))
    def isOverflow(point, width, height):
        if point[0] < 0 or point[1] < 0 or point[0] > width - 1 or point[1] > height - 1:
            return True
        return False

    if isOverflow(startPoint, width, height) or isOverflow(endPoint, width, height) or isOverflow(centerPoint, width,
                                                                                                  height):
        # print("overflow!")
        return [template,0]

    # startPointUp = (int(round(newpoints[7][0]) - newpoints[0][0]), int(round(newpoints[7][1]) - newpoints[0][1]))
    # endPointUp = (int(round(newpoints[8][0]) - newpoints[0][0]), int(round(newpoints[8][1]) - newpoints[0][1]))
    # centerPointUp = (int(round(newpoints[9][0]) - newpoints[0][0]), int(round(newpoints[9][1]) - newpoints[0][1]))
    #todo 将旋转后的点的坐标 存到json文件
    info["startPoint"]["x"] = startPoint[0]
    info["startPoint"]["y"] = startPoint[1]
    info["centerPoint"]["x"] = centerPoint[0]
    info["centerPoint"]["y"] = centerPoint[1]
    info["endPoint"]["x"] = endPoint[0]
    info["endPoint"]["y"] = endPoint[1]
    info["charastartPoint"]["x"] = charastartPoint[0]
    info["charastartPoint"]["y"] = charastartPoint[1]
    info["charaendPoint"]["x"] = charaendPoint[0]
    info["charaendPoint"]["y"] = charaendPoint[1]
    info["characenterPoint"]["x"] = characenterPoint[0]
    info["characenterPoint"]["y"] = characenterPoint[1]

    return [src_correct,1]

# def meterReginAndLocationBySIFT(orign_img, info):
#     """
#     locate meter's bbox
#     :param orign_img: image
#     :param info: info
#     :return: bbox image and bbox
#     """
#     if str(info["type"])[10:13]!='vid' or str(info["type"])[10:13]!='dig':
#         x = info["ROI"]["x"]
#         y = info["ROI"]["y"]
#         w = info["ROI"]["w"]
#         h = info["ROI"]["h"]
#         # call back
#         # cv2.rectangle(recognitionData, (x, y), (x+w, y + h), (255, 0, 0), 3)
#         # cv2.imshow("testInput", recognitionData)
#         # cv2.waitKey(0)
#         if x != 0 or y != 0 or w != 0 or h != 0:
#             image = orign_img[y:y + h, x:x + w]
#         else:
#             image = orign_img
#     template = info["template"]
#
#     # cv2.imshow("template", template)
#     # cv2.imshow("image", image)
#     # cv2.waitKey(0)
#
#     startPoint = (info["startPoint"]["x"], info["startPoint"]["y"])
#     centerPoint = (info["centerPoint"]["x"], info["centerPoint"]["y"])
#     endPoint = (info["endPoint"]["x"], info["endPoint"]["y"])
#     # startPointUp = (info["startPointUp"]["x"], info["startPointUp"]["y"])
#     # endPointUp = (info["endPointUp"]["x"], info["endPointUp"]["y"])
#     # centerPointUp = (info["centerPointUp"]["x"], info["centerPointUp"]["y"])
#
#     templateBlurred = cv2.GaussianBlur(template, (3, 3), 0)
#     imageBlurred = cv2.GaussianBlur(image, (3, 3), 0)
#
#     sift = cv2.xfeatures2d.SIFT_create()
#
#     # shape of descriptor n * 128, n is the num of key points.
#     # a row of descriptor is the feature of related key point.
#     templateKeyPoint, templateDescriptor = sift.detectAndCompute(templateBlurred, None)
#     imageKeyPoint, imageDescriptor = sift.detectAndCompute(imageBlurred, None)
#
#     # for debug
#     # templateBlurred = cv2.drawKeypoints(templateBlurred, templateKeyPoint, templateBlurred)
#     # imageBlurred = cv2.drawKeypoints(imageBlurred, imageKeyPoint, imageBlurred)
#     # cv2.imshow("template", templateBlurred)
#     # cv2.imshow("image", imageBlurred)
#
#     # match
#     bf = cv2.BFMatcher()
#     # k = 2, so each match has 2 points. 2 points are sorted by distance.
#     matches = bf.knnMatch(templateDescriptor, imageDescriptor, k=2)
#
#     # The first one is better than the second one
#     good = [[m] for m, n in matches if m.distance < 0.8 * n.distance]
#
#     # distance matrix
#     templatePointMatrix = np.array([list(templateKeyPoint[p[0].queryIdx].pt) for p in good])
#     imagePointMatrix = np.array([list(imageKeyPoint[p[0].trainIdx].pt) for p in good])
#     templatePointDistanceMatrix = pairwise_distances(templatePointMatrix, metric="euclidean")
#     imagePointDistanceMatrix = pairwise_distances(imagePointMatrix, metric="euclidean")
#
#     # del bad match
#     distances = []
#     maxAbnormalNum = 15
#     for i in range(len(good)):
#         diff = abs(templatePointDistanceMatrix[i] - imagePointDistanceMatrix[i])
#         # distance between distance features
#         diff.sort()
#         distances.append(np.sqrt(np.sum(np.square(diff[:-maxAbnormalNum]))))
#
#     averageDistance = np.average(distances)
#     good2 = [good[i] for i in range(len(good)) if distances[i] < 2 * averageDistance]
#
#     # for debug
#     # matchImage = cv2.drawMatchesKnn(template, templateKeyPoint, image, imageKeyPoint, good2, None, flags=2)
#     # cv2.imshow("matchImage", matchImage)
#     # cv2.waitKey(0)
#
#     # not match
#     if len(good2) < 3:
#         # print("not found!")
#         return template, None, None,0
#
#     # 寻找转换矩阵 M
#     src_pts = np.float32([templateKeyPoint[m[0].queryIdx].pt for m in good2]).reshape(-1, 1, 2)
#     dst_pts = np.float32([imageKeyPoint[m[0].trainIdx].pt for m in good2]).reshape(-1, 1, 2)
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#     h, w, _ = template.shape
#
#     # 找出匹配到的图形的四个点和标定信息里的所有点
#     pts = np.float32(
#         [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0], [startPoint[0], startPoint[1]], [endPoint[0], endPoint[1]],
#          [centerPoint[0], centerPoint[1]],
#          # [startPointUp[0], startPointUp[1]],
#          # [endPointUp[0], endPointUp[1]],
#          # [centerPointUp[0], centerPointUp[1]]
#          ]).reshape(-1, 1, 2)
#     dst = cv2.perspectiveTransform(pts, M)
#
#     # 校正图像
#     angle = 0.0
#     vector = (dst[3][0][0] - dst[0][0][0], dst[3][0][1] - dst[0][0][1])
#     cos = (vector[0] * (200.0)) / (200.0 * math.sqrt(vector[0] ** 2 + vector[1] ** 2))
#     if (vector[1] > 0):
#         angle = math.acos(cos) * 180.0 / math.pi
#     else:
#         angle = -math.acos(cos) * 180.0 / math.pi
#     # print(angle)
#
#     change = cv2.getRotationMatrix2D((dst[0][0][0], dst[0][0][1]), angle, 1)
#     src_correct = cv2.warpAffine(image, change, (image.shape[1], image.shape[0]))
#     array = np.array([[0, 0, 1]])
#     newchange = np.vstack((change, array))
#     # 获得校正后的所需要的点
#     newpoints = []
#     for i in range(len(pts)):
#         point = newchange.dot(np.array([dst[i][0][0], dst[i][0][1], 1]))
#         point = list(point)
#         point.pop()
#         newpoints.append(point)
#     src_correct = src_correct[int(round(newpoints[0][1])):int(round(newpoints[1][1])),
#                   int(round(newpoints[0][0])):int(round(newpoints[3][0]))]
#     bbox = [int(round(newpoints[0][1])), int(round(newpoints[1][1])),
#                   int(round(newpoints[0][0])), int(round(newpoints[3][0]))]
#
#
#     width = src_correct.shape[1]
#     height = src_correct.shape[0]
#     if width == 0 or height == 0:
#         return template, None, None,0
#
#     startPoint = (int(round(newpoints[4][0]) - newpoints[0][0]), int(round(newpoints[4][1]) - newpoints[0][1]))
#     endPoint = (int(round(newpoints[5][0]) - newpoints[0][0]), int(round(newpoints[5][1]) - newpoints[0][1]))
#     centerPoint = (int(round(newpoints[6][0]) - newpoints[0][0]), int(round(newpoints[6][1]) - newpoints[0][1]))
#
#     def isOverflow(point, width, height):
#         if point[0] < 0 or point[1] < 0 or point[0] > width - 1 or point[1] > height - 1:
#             return True
#         return False
#
#     if isOverflow(startPoint, width, height) or isOverflow(endPoint, width, height) or isOverflow(centerPoint, width,
#                                                                                                   height):
#         # print("overflow!")
#         return template, None, None,0
#
#     info["startPoint"]["x"] = startPoint[0]
#     info["startPoint"]["y"] = startPoint[1]
#     info["centerPoint"]["x"] = centerPoint[0]
#     info["centerPoint"]["y"] = centerPoint[1]
#     info["endPoint"]["x"] = endPoint[0]
#     info["endPoint"]["y"] = endPoint[1]
#     return src_correct, change, bbox,1
