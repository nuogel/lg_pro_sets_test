def lengthOfLIS(nums) -> int:
    output = []
    count = []
    for i in range(len(nums)):
        output.append(1)
        count.append(1)
        for j in range(i):
            if nums[i] > nums[j]:
                if output[j] + 1 > output[i]:
                    count[i] =count[j]
                elif output[j] + 1 == output[i]:
                    count[i] += count[j]
                output[i] = max(output[i], output[j] + 1)
    result = 0
    maxi = max(output)
    for i in range(len(output)):
        if maxi == output[i]:
            result += count[i]
    return result


lengthOfLIS([1, 2, 4, 3, 5, 4, 7, 2])
