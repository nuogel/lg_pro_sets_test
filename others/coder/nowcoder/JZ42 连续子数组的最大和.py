#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
#
# @param array int整型一维数组
# @return int整型
#
import numpy as np


class Solution:
    def FindGreatestSumOfSubArray_1(self, array) -> int:
        # write code here
        summax = array[0]
        max_ = [summax]
        for i in range(1, len(array)):
            if max_[-1] >= 0:
                summax = summax + array[i]
                max_.append(summax)
            else:
                summax = array[i]
                max_.append(array[i])
        maxone = max(max_)
        return maxone

    def FindGreatestSumOfSubArray_2(self, array) -> int:
        # write code here
        dpi = [array[0]]
        for i in range(1, len(array)):
            dpi.append(max(array[i], dpi[-1]+array[i]))
        maxone = max(dpi)
        return maxone


if __name__ == '__main__':
    array = [1, -2, 3, 10, -4, 7, 2, -5]
    s = Solution()
    y = s.FindGreatestSumOfSubArray_2(array)
    print(y)
