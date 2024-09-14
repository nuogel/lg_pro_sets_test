def findMin(nums) -> int:
    # return min(nums)
    left=0
    right=len(nums)-1
    while left<right:
        mid = left+(right-left) // 2
        if nums[mid]>nums[right]:
            left=mid+1
        else:
            right=mid
    return nums[left]


print(findMin([3,4,5,1,2]))