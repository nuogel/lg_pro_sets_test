'''
给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。



示例 1：

输入：nums = [10,2]
输出："210"

示例 2：

输入：nums = [3,30,34,5,9]
输出："9534330"

'''


from functools import cmp_to_key

def largestNumber(nums) -> str:
    # def compare(x,y):
    #     if int(x+y)>int(y+x):
    #         return x
    #     else:
    #         return y

    # sorted(nums, key=lambda x:y, compare(x,y))

    def compare(x, y): return int(y + x) - int(x + y)

    nums = sorted(map(str, nums), key=cmp_to_key(compare))
    return "0" if nums[0] == "0" else "".join(nums)


largestNumber([0,0])

