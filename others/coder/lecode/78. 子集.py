def subsets(nums):

    output = []
    nums.sort()
    def solve(i, path):
        output.append(path[:])
            # return
        for ni in range(i, len(nums)):
            if nums[ni] in path: continue
            path.append(nums[ni])
            solve(ni+1, path)
            path.pop()

    solve(0, [])
    return output

subsets([3,2,4,1])