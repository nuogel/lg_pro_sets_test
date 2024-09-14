def combinationSum(candidates, target: int):
    output = []

    def tackback(path):
        if sum(path) == target:
            path=list(path)
            path.sort()
            if path not in output:
                output.append(list(path))
            return
        elif sum(path)>target:
            return
        for i in range(len(candidates)):
            path.append(candidates[i])
            tackback(path)
            path.pop()

    tackback([])
    return output

combinationSum([1,2],4)