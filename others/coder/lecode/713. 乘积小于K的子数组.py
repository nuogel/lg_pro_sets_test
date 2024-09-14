def numSubarrayProductLessThanK(nums, k: int) -> int:
    if k <= 0: return 0
    l = 0
    r = 0
    prod = 1
    output = 0
    while l <= r < len(nums):
        prod *= nums[r]
        while prod >= k and l <= r < len(nums):
            prod = prod / nums[l]
            l += 1
        output += r - l + 1
        r += 1
    return output


numSubarrayProductLessThanK([10,5,2,6],100)