'''
同最大路径和
'''

class Solution:
    def maxValue(self, grid) -> int:
        # write code here
        m, n = len(grid), len(grid[0])
        maxgrid = [[0 for i in range(n + 1)] for j in range(m+ 1)]

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                maxgrid[i][j] = max(maxgrid[i + 1][j], maxgrid[i][j + 1]) + grid[i][j]

        return maxgrid[0][0]


if __name__ == '__main__':
    grid = [[9, 1, 4, 8]]
    s = Solution()
    s.maxValue(grid)


def tryit(grid):
    m = len(grid)
    n = len(grid[0])

    for i in range(1,m):
        grid[i][0] += grid[i-1][0]

    for i in range(n):
        grid[0][i] += grid[0][i-1]

    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += max(grid[i-1][j], grid[i][j-1])

    return grid[-1][-1]




