#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
#
# @param array int整型一维数组
# @return int整型
#
import numpy as np


class Solution:
    def FindGreatestSumOfSubArray(self, array) -> int:
        # write code here
        dpi = [array[0]]
        s, e = 0, 1
        maxs, maxe = 0, 1
        maxdpi = array[0]
        for i in range(1, len(array)):
            dpi.append(max(array[i], dpi[i - 1] + array[i]))
            if dpi[i - 1] < 0:  # 先记录起点
                s = i
            if dpi[-1] >= maxdpi:  # 当从起点到下一点的和>最大值的时候，记录终点。
                maxs = s
                maxe = i + 1
                maxdpi = dpi[-1]
        return array[maxs:maxe]


if __name__ == '__main__':
    array = [1, -2, 3, 10, -4, 7, 2, -5]  # [-1, 3, 2, 1, -2, -2, -3, 0, 3, 2, 1, -1]  # [-2, -8, -1, -5, -9]  #
    s = Solution()
    y = s.FindGreatestSumOfSubArray(array)
    print(y)
