nums = [-1, 0, 0, 0, 0, 3, 3]
# a = list(set(nums))
# nums[:len(a)] = a
# print(a)

nums = [3, 2, 2, 3]
val = 3
left = 0
right = 0

ll = len(nums)
while right < ll:
    if nums[right] != val:
        nums[left] = nums[right]
        left += 1
    right += 1
a = 0
