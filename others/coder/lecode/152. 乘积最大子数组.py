def maxProduct(nums) -> int:
    maxv = [nums[0]]
    minv = [nums[0]]

    for ni in nums[1:]:
       maxvi = max(max(ni, maxv[-1]*ni), max(ni, minv[-1]*ni))
       minvi = min(min(ni, maxv[-1]*ni), min(ni, minv[-1]*ni))
       maxv.append(maxvi)
       minv.append(minvi)

    return max(maxv)

maxProduct([-4,-3,-2])


def tyrit(nums):
    minv = [nums[0]]
    maxv = [nums[0]]

    for i in range(1, len(nums)):
        maxi = max(max(maxv[-1] * nums[i], nums[i]), max(minv[-1] * nums[i], nums[i]))
        mini = min(min(maxv[-1] * nums[i], nums[i]), min(minv[-1] * nums[i], nums[i]))
        maxv.append(maxi)
        minv.append(mini)
    return max(maxv)

tyrit([-4,-3,-2])