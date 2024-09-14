def threeSum( nums):
    # ll=len(nums)
    # out=[]
    # for i in range(ll):
    #     for  j in range(i+1, ll):
    #         for k in range(j+1, ll):
    #             if nums[i]+nums[j]+nums[k]==0:
    #                 outi= [nums[i],nums[j],nums[k]]
    #                 outi.sort()
    #                 if outi not in out:
    #                     out.append(outi)

    # return out
    # nums=list(set(nums))
    if len(nums) < 3:
        return []

    def twosum(nums2, sum2, others):
        out = []
        s = 0
        e = len(nums2) - 1
        while s < e:
            if nums2[s] + nums2[e] < sum2:
                s += 1
            elif nums2[s] + nums2[e] > sum2:
                e -= 1
            else:
                out.append([nums2[s], nums2[e]] + others)
                s += 1
                e -= 1
        return out

    output = []
    nums.sort()
    for i in range(len(nums)):
        if i + 1 < len(nums):
            ts = twosum(nums[i + 1:], -nums[i], [nums[i]])
            if ts != []:
                for tsi in ts:
                    if tsi not in output:
                        output.append(tsi)
    return output

threeSum([-1,0,1,2,-1,-4])