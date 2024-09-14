def longestCommonSubsequence(text1: str, text2: str) -> int:

    l1=len(text1)
    l2=len(text2)
    matrix = [[0 for i in range(l2+1)] for i in range(l1+1)]

    for i in range(l1):
        for j in range(l2):
            if text1[i]==text2[j]:
                matrix[i+1][j+1]=matrix[i][j]+1
            else:
                matrix[i + 1][j + 1]=max(matrix[i+1][j], matrix[i][j+1])
    return matrix[-1][-1]

longestCommonSubsequence('abcde', 'ace')