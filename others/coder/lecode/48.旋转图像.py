from typing import List


def rotate(matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    n = len(matrix)

    copy = [[0 for i in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(n):
            copy[j][n - 1 - i] = matrix[i][j]

    return copy


rotate([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
