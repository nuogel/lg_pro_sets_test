import math

def maximalRectangle(matrix) -> int:
    m = len(matrix)
    n = len(matrix[0])
    maxa = 0

    def largestRectangleArea3(heights) -> int:
        starck = [0]
        heights.insert(0, 0)
        heights.append(0)
        id = 0
        areamax = 0
        while id < len(heights):
            while id < len(heights) and heights[starck[-1]] <= heights[id]:
                starck.append(id)
                id += 1

            while id < len(heights) and heights[starck[-1]] > heights[id]:
                ipop = starck.pop()
                right = id - 1
                left = starck[-1] + 1
                areamax = max(areamax, (right - left + 1) * heights[ipop])
        return areamax

    for i in range(m):
        for j in range(n):
            matrix[i][j] = int(matrix[i][j])
            if i - 1 >= 0 and matrix[i][j] != 0:
                matrix[i][j] = matrix[i - 1][j]+matrix[i][j]

        mi = matrix[i]
        maxa = max(maxa, largestRectangleArea3(mi.copy()))

    return maxa


maximalRectangle([["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"], ["1", "0", "0", "1", "0"]])
