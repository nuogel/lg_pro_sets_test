def convert(s: str, numRows: int) -> str:
    if numRows == 1: return s
    ll = len(s)
    clo = ll
    matrix = [['' for i in range(clo)] for j in range(numRows)]
    r = 0
    i = 0
    c = 0
    while i < len(s):
        while r < numRows and i < len(s):
            matrix[r][c] = s[i]
            r += 1
            i += 1
        c += 1
        r -= 2
        while r > 0 and i < len(s):
            matrix[r][c] = s[i]
            r -= 1
            c += 1
            i += 1
    out = ''
    for i in range(numRows):
        for j in range(clo):
            print(matrix[i][j])
            if matrix[i][j] != '':
                out += matrix[i][j]
    return out


convert("ABC", 1)
