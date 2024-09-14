'''
给定一个长度为偶数的整数数组 arr，只有对 arr 进行重组后可以满足 “对于每个 0 <= i < len(arr) / 2，都有 arr[2 * i + 1] = 2 * arr[2 * i]” 时，返回 true；否则，返回 false。

示例 1：

输入：arr = [3,1,3,6]
输出：false

示例 2：

输入：arr = [2,1,2,6]
输出：false

示例 3：

输入：arr = [4,-2,2,-4]
输出：true
解释：可以用 [-2,-4] 和 [2,4] 这两组组成 [-2,-4,2,4] 或是 [2,4,-2,-4]

'''

def solve(arr):
    arrset = list(set(arr))
    arrset.sort()
    if 0 in arrset:
        arrset.remove(0)
    arrdict = {}
    for arri in arrset:
        arrdict[arri] = arr.count(arri)

    for arri in  arrset:
        if arrdict[arri] == 0:
            continue
        arr2 = [arri*2, arri/2]
        for arr2i in arr2:
            if arr2i in arrset:
                mincount = min(arrdict[arri], arrdict[arr2i])
                arrdict[arri] -= mincount
                arrdict[arr2i] -= mincount

    for k,v in arrdict.items():
        if v!=0:
            return False
    return True


solve([-1,4,6,8,-4,6,-6,3,-2,3,-3,-8])