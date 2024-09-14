def solve(nums):
    ll = len(nums)
    value = []
    for i in range(ll):
        if i<=1:
            value.append(max(nums[:i+1]))
        else:
            maxv = max(nums[i]+value[i-2], value[i-1])
            value.append(maxv)

    return max(value)


solve([2,7,9,3,1])
