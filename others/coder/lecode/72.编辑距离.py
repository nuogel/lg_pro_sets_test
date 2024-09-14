def minDistance(word1: str, word2: str) -> int:
    l1 = len(word1)
    l2 = len(word2)

    edit = [[0 for i in range(l2 + 1)] for j in range(l1+ 1)]

    for i in range(l1 + 1):
        edit[i][0] = i
    for i in range(l2 + 1):
        edit[0][i] = i

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if word1[i-1] == word2[j-1]:
                edit[i][j] = edit[i - 1][j - 1]
            else:
                edit[i][j] = min(edit[i - 1][j - 1], edit[i - 1][j] , edit[i][j - 1])+1
    return edit[l1][l2]

minDistance('horse', 'ros')