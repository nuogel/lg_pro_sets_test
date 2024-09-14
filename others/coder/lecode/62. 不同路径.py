def uniquePaths(m: int, n: int) -> int:
    martix = [[0 for i in range(n + 1)] for j in range(m + 1)]
    martix[m - 1][n - 1] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if martix[i][j] == 0:
                martix[i][j] = martix[i + 1][j] + martix[i][j + 1]
    return martix[0][0]


global step
step = 0


def uniquePaths2(m: int, n: int) -> int:
    martix = [[-1 for i in range(n)] for j in range(m)]

    def uinq(m, n):
        if m < 0 or n < 0:
            return 0
        if martix[m][n] != -1:
            return 0
        martix[m][n] = 1
        if m == 0 and n == 0:
            return 1
        return uniquePaths2(m - 1, n) + uniquePaths2(m, n - 1)

    out = uinq(m - 1, n - 1)
    return out

def uniquePaths3(m: int, n: int) -> int:
    martix = [[-1 for i in range(n)] for j in range(m)]

    for i in range(m):
        martix[i][0]=1
    for j in range(n):
        martix[0][j]=1

    for i in range(1, m):
        for j in range(1, n):
            martix[i][j] = martix[i-1][j]+martix[i][j-1]

    return  martix[-1][-1]



uniquePaths(3, 7)
print(uniquePaths2(3, 7))
print(uniquePaths3(3, 7))
