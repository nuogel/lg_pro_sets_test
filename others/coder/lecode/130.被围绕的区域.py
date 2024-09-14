def solve(board) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    m = len(board)
    n = len(board[0])

    steped = [[0 for i in range(n)] for i in range(m)]

    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and board[i][j] == 'O' and steped[i][j] == 0:
            steped[i][j] = 1
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)

    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)

    for i in range(m):
        for j in range(n):
            if steped[i][j] == 0 and board[i][j] == 'O':
                board[i][j] = 'X'
    return board


solve([["X","O","X","O","X","O"],["O","X","O","X","O","X"],["X","O","X","O","X","O"],["O","X","O","X","O","X"]])