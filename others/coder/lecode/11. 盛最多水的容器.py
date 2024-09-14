def solve(height):
    l = 0
    r = len(height) - 1
    maxw = 0
    while l < r:
        maxw = max(maxw, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return maxw


solve([1,8,6,2,5,4,8,3,7])