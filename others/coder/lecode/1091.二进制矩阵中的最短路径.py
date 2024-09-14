def shortestPathBinaryMatrix(grid) -> int:
    m = len(grid)
    n = len(grid[0])
    steped = [[0] * m for i in range(n)]
    if grid[0][0]==1:
        return -1
    q = [(0, 0)]
    steped[0][0]=1
    step = 1
    while q:
        ll=len(q)
        for li in range(ll):
            i,j = q.pop(0)
            if i == m - 1 and j == n - 1:
                return step
            for ii, jj in [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1), (i, j - 1), (i, j + 1)]:
                if 0 <= ii < m and 0 <= jj < n and steped[ii][jj] == 0 and grid[ii][jj]!=1:
                    if ii==m-1 and jj==n-1:
                        return step+1
                    q.append((ii,jj))
                    steped[ii][jj]=1
        step += 1
    return -1


shortestPathBinaryMatrix([[0]]

)
