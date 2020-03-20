import numpy as np


class AngleFactory:
    """method for angle calculation"""

    @staticmethod
    def __calAngleBetweenTwoVector(vectorA, vectorB):  # todo 根据向量 计算角度
        """
        get angle formed by two vector
        :param vectorA: vector A
        :param vectorB: vector B
        :return: angle
        """
        lenA = np.sqrt(vectorA.dot(vectorA))
        lenB = np.sqrt(vectorB.dot(vectorB))
        cosAngle = vectorA.dot(vectorB) / (lenA * lenB)
        angle = np.arccos(cosAngle)
        return angle

    @classmethod
    def calAngleClockwise(cls, startPoint, endPoint, centerPoint):  # todo 根据 起点，终点，中心点 计算角度
        """
        get clockwise angle formed by three point
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :return: clockwise angle
        """
        vectorA = startPoint - centerPoint
        vectorB = endPoint - centerPoint
        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)

        # if counter-clockwise
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        return angle

    @classmethod
    def calPointerValueByOuterPoint(cls, startPoint, endPoint, centerPoint, pointerPoint, startValue, totalValue):
        """
        # todo      计算读数
        get value of pointer meter
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :param pointerPoint: pointer's outer point  指针的顶点坐标  也就是靠近量程刻线的那一端
        :param startValue: start value
        :param totalValue: total value
        :return: value
        """
        # print(startPoint, endPoint, centerPoint, pointerPoint, startValue, totalValue)
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)  # todo 计算量程所张开的角度
        angle = cls.calAngleClockwise(startPoint, pointerPoint, centerPoint)   # todo 计算指针与量程一端所张开的角度
        value = angle / angleRange * (totalValue - startValue) + startValue
        if value > totalValue or value < startValue:  # todo 只是防止读数错误
            return startValue if angle > np.pi + angleRange / 2 else totalValue
        return value

    @classmethod
    def calPointerValueByPointerVector(cls, startPoint, endPoint, centerPoint, PointerVector, startValue, totalValue):
        """
        # todo 也是计算 读数，这个 函数没用到
        get value of pointer meter
        注意传入相对圆心的向量
        :param startPoint: start point
        :param endPoint: end point
        :param centerPoint: center point
        :param PointerVector: pointer's vector
        :param startValue: start value
        :param totalValue: total value
        :return: value
        """
        angleRange = cls.calAngleClockwise(startPoint, endPoint, centerPoint)

        vectorA = startPoint - centerPoint
        vectorB = PointerVector

        angle = cls.__calAngleBetweenTwoVector(vectorA, vectorB)

        # if counter-clockwise
        if np.cross(vectorA, vectorB) < 0:
            angle = 2 * np.pi - angle

        value = angle / angleRange * totalValue + startValue
        if value > totalValue or value < startValue:
            return startValue if angle > np.pi + angleRange / 2 else totalValue
        return value

    # def findPointerFromHSVSpace(src, center, radius, radians_low, radians_high, patch_degree=1.0, ptr_resolution=5,
    #                             low_ptr_color=np.array([0, 0, 221]), up_ptr_color=np.array([180, 30, 255])):
    #
    #     """
    #     从固定颜色的区域找指针,未完成
    #     :param low_ptr_color: 指针的hsv颜色空间的下界
    #     :param up_ptr_color:  指针的hsv颜色空间的上界
    #     :param radians_low:圆的搜索范围(弧度制表示)
    #     :param radians_high:圆的搜索范围(弧度制表示)
    #     :param src: 二值图
    #     :param center: 刻度盘的圆心
    #     :param radius: 圆的半径
    #     :param patch_degree:搜索梯度，默认每次一度
    #     :param ptr_resolution: 指针的粗细程度
    #     :return: 指针遮罩、直线与圆相交的点
    #     """
    #
    # pass
