def findCircleNum(isConnected) -> int:
    m = len(isConnected)
    visited = [0] * m
    output = 0

    def dfs(i):
        visited[i] = 1
        for j in range(m):
            if visited[j] == 0 and isConnected[i][j] == 1:
                dfs(j)

    for i in range(m):
        if visited[i] == 0:
            output += 1
            dfs(i)

    return output


findCircleNum([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]])
