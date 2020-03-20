from Algorithm.utils.Finder import meterFinderBySIFT
from Algorithm.utils.ScanPointer import scanPointer
 # todo 指针表
import copy
def normalPressure(image, allinfo):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """
    template = None
    flag = None
    info = None
    if type(allinfo) is list:
        for i in range(len(allinfo)): # todo 用传入的 多个模板 进行一个个的匹配，有一个匹配到则跳出循环
            info = allinfo[i]

            template, flag = meterFinderBySIFT(image, info)  # todo flag==0 则是没有匹配到，继续用下一个模板匹配
            if flag == 0:
                continue
            else:
                break

    else:
        template, flag = meterFinderBySIFT(image, allinfo)
        info = copy.deepcopy(allinfo)
    if flag == 0:
        print('not find template!!!')

    result = scanPointer(template, info)  # todo 调用 指针 识别函数
    result = int(result * 1000) / 1000
    return [result]

