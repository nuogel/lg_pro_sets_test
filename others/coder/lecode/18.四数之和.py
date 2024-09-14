def fourSum(nums, target: int):
    # output = []
    # lastpop = 10000000
    # nums.sort()
    #
    # def sovle(ii, path, ids, lastpop):
    #     if len(path) == 4 and sum(path) == target:
    #         pathi = path[:]
    #         pathi.sort()
    #         if pathi not in output:
    #             output.append(pathi)
    #             return
    #     elif len(path) > 4:
    #         return
    #
    #     for i in range(ii, len(nums)):
    #         if i in ids or nums[i] == lastpop: continue
    #         path.append(nums[i])
    #         ids.append(i)
    #         sovle(ii, path, ids, lastpop)
    #         lastpop = path.pop()
    #         ids.pop()
    #
    # sovle(0, [], [], lastpop)
    # return output
    n = len(nums)
    out = []
    nums.sort()
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        if sum(nums[i:i + 4]) > target:
            break
        if nums[i] + sum(nums[n - 3:]) < target:
            continue

        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            if nums[i] + sum(nums[j:j + 3]) > target:
                break
            if nums[i] + nums[j] + sum(nums[n - 2:]) < target:
                continue
            for m in range(j + 1, n - 1):
                for k in range(m + 1, n):
                    if nums[i] + nums[j] + nums[m] + nums[k] == target:
                        re = [nums[i], nums[j], nums[m], nums[k]]
                        re.sort()
                        if re not in out:
                            out.append(re)
    return out


fourSum([1,0,-1,0,-2,2],0)