import collections

import numpy as np
import math
import copy


# 冒泡排序
def bubble_sort(nums):
    ll = len(nums)
    for i in range(ll):
        for j in range(i + 1, ll):
            if nums[i] > nums[j]:
                nums[i], nums[j] = nums[j], nums[i]
    return nums


# 插入排序

def insert_sort(nums):
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] < nums[j]:
                numi = nums[i]
                nums = nums[:i] + nums[i + 1:]
                nums.insert(j, numi)
    return nums


# 归并排序
def merge_sort(alist):
    # 合并
    def merge(left, right):
        '''合并操作，将两个有序数组left[]和right[]合并成一个大的有序数组'''
        # left与right的下标指针
        l, r = 0, 0
        result = []
        while l < len(left) and r < len(right):
            if left[l] < right[r]:
                result.append(left[l])
                l += 1
            else:
                result.append(right[r])
                r += 1
        result += left[l:]
        result += right[r:]
        return result

    if len(alist) <= 1:
        return alist
    # 二分分解
    num = len(alist) // 2
    left = merge_sort(alist[:num])
    right = merge_sort(alist[num:])

    return merge(left, right)


# 快速排序
def quick_sort(arr, l, r):
    def partition(arr, l, r):
        x = arr[r]
        i = l - 1
        for j in range(l, r):
            if arr[j] <= x:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[r] = arr[r], arr[i + 1]
        return i + 1

    if l < r:
        q = partition(arr, l, r)
        quick_sort(arr, l, q - 1)
        quick_sort(arr, q + 1, r)
    return arr


def quick_lg_1(arr, l, r):
    def partion_lg(arr, l, r):
        x = arr[l]
        out = [x]
        index = l
        for arri in arr[l + 1:r + 1]:
            if arri <= x:
                out.insert(0, arri)
                index += 1
            else:
                out.append(arri)
        arr[l:r + 1] = out
        return index, arr

    if l < r and l >= 0 and r <= len(arr):
        index, arr = partion_lg(arr, l, r)
        quick_lg_1(arr, l, index - 1)
        quick_lg_1(arr, index + 1, r)
    return arr


def quick_lg_2(arr):
    def partion_lg(arr):
        x = arr[0]
        left = []
        right = []
        for arri in arr[1:]:
            if arri <= x:
                left.insert(0, arri)
            else:
                right.append(arri)
        return left, right, [x]

    if len(arr) > 1:
        left, right, x = partion_lg(arr)
        left = quick_lg_2(left)
        right = quick_lg_2(right)
        arr = left + x + right
    return arr


# def b_search(arr, left, right, target):
#     if left <= right:
#         mid = int(left + (right - left) / 2)
#         if arr[mid] == target:
#             return mid, target
#         elif arr[mid] < target:
#             return b_search(arr, mid + 1, right, target)
#         else:
#             return b_search(arr, left, mid - 1, target)
#     else:
#         return None


# 希尔排序
def shellsort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            tmpi = arr[i]
            while i >= gap and arr[i - gap] > tmpi:
                arr[i], arr[i - gap] = arr[i - gap], arr[i]
                i -= gap
        gap = gap // 2
    return arr


# 计数排序
def countingSort(arr):
    _max = max(arr)
    _min = min(arr)

    n = len(arr)
    c = [0] * (_max - _min + 1)
    output = [0] * n
    for arri in arr:
        c[arri - _min] += 1

    for i in range(1, (_max - _min + 1)):
        c[i] = c[i - 1] + c[i]

    for arri in arr:
        output[c[arri - _min] - 1] = arri
        c[arri - _min] -= 1

    return output


# 计数排序 好理解
def count_sort(arr):  # arr为待排序列表，max_count为最大元素
    _max = max(arr)
    _min = min(arr)
    count = [0 for _ in range(_max - _min + 1)]  # 列表推导式生成0到500的列表，用来记录元素出现多少次
    for val in arr:
        count[val - _min] += 1  # 如果元素出现则对应count列表索引处+1
    arr.clear()  # 直接清除原列表，不在生成新列表，节省内容空间
    for index, val in enumerate(count):  # 获取index下标，val对应的值
        # for循环里， index索引出现了val次
        for i in range(val):
            arr.append(index + _min)  # 把index添加到arr，次数为val
    return arr

def count_sort2(nums):
    ids = list(set(nums))
    ids.sort()
    dictnum = {}
    for idsi in ids:
        dictnum[idsi]=0
    for ni in nums:
        dictnum[ni]+=1
    output = []
    for k,v in dictnum.items():
        output+=[k]*v
    return output


if __name__ == '__main__':
    arr = list(np.random.randint(0, 20, 10))
    print('raw_arr:', arr)
    arrc = copy.copy(arr)
    arrc.sort()
    print('arr.sort():', arrc)
    y = bubble_sort(copy.copy(arr))
    print(y)
    y = insert_sort(copy.copy(arr))
    print(y)
    y = merge_sort(copy.copy(arr))
    print(y)
    y = shellsort(copy.copy(arr))
    print(y)
    y = countingSort(copy.copy(arr))
    print(y)
    y = count_sort(copy.copy(arr))
    print(y)
    y = count_sort2(copy.copy(arr))
    print(y)
    y = quick_sort(copy.copy(arr), 0, len(copy.copy(arr)) - 1)
    print(y)
    y = quick_lg_1(copy.copy(arr), 0, len(copy.copy(arr)) - 1)
    print(y)
    y = quick_lg_2(copy.copy(arr))
    print(y)
