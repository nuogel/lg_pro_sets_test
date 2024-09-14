def jump(nums) -> int:
    if len(nums) == 1:
        return 0

    minstep = 0
    step = 0
    while step < len(nums)-1:
        maxdis = 0
        for i in range(step+1, step+nums[step]+1):
            if i >= len(nums) - 1:
                return minstep+1
            elif nums[i]+i >= maxdis:
                maxdis=i+nums[i]
                step=i
        minstep += 1
    return minstep


jump([1,2,1,1,1])