import copy
class Solution:
    def highestPeak(self, isWater):
        m = len(isWater)
        n = len(isWater[0])
        mount = [[1000000 for i in range(n)] for j in range(m)]
        visited = [[0 for i in range(n)] for j in range(m)]

        def makemount(i, j, visitedcopy):
            if not (0 <= i < m and 0 <= j < n) or visitedcopy[i][j]:
                return
            if isWater[i][j] == 1:
                mount[i][j] = 0
            else:
                minm = 1000000
                if 0 <= i - 1 < m:
                    minm = min(mount[i - 1][j], minm)
                if i + 1 < m:
                    minm = min(mount[i + 1][j], minm)
                if 0 <= j - 1 < n:
                    minm = min(mount[i][j - 1], minm)
                if j + 1 < n:
                    minm = min(mount[i][j + 1], minm)
                mount[i][j] = minm + 1

            visitedcopy[i][j] = 1
            for (di, dj) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if  (0 <= di < m and 0 <= dj < n) and visitedcopy[di][dj] == 0:
                    makemount(di, dj, visitedcopy)

        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    visitedcopy = copy.deepcopy(visited)
                    makemount(i, j, visitedcopy)

        return mount


if __name__ == '__main__':
    wa = [[0, 0, 1], [1, 0, 0], [0, 0, 0]]
    s = Solution()
    print(s.highestPeak(wa))
