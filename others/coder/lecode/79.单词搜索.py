def exist(board, word: str) -> bool:
    m = len(board)
    n = len(board[0])

    def solve(idx, i, j, path, steped):
        if idx == len(word):
            return True
        for (ii, jj) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            if 0 <= ii < m and 0 <= jj < n and (ii, jj) not in steped and word[idx] == board[ii][jj]:
                path.append(board[ii][jj])
                steped.append((ii, jj))
                if solve(idx + 1, ii, jj, path, steped):
                    return True
                path.pop()
                steped.pop()
    for i in range(m):
        for j in range(n):
            if word[0] == board[i][j]:
                if solve(1, i, j, [word[0]], [(i,j)]):
                    return True
    return False

exist([["a","a"]],"aaa")
