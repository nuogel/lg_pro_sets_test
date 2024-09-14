def minSubArrayLen(target: int, nums) -> int:
    if target <= 0: return 0
    l = 0
    r = 0
    sumall = 0
    minlenth = len(nums)+1
    while l <= r < len(nums):
        sumall += nums[r]
        while sumall >= target and l<=r:
            minlenth = min(minlenth, r - l+1)
            sumall -= nums[l]
            l+= 1
        r += 1
    if minlenth==len(nums)+1:
        minlenth=0
    return minlenth

minSubArrayLen(7,[2,3,1,2,4,3])