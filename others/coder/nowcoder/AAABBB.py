def winnerOfGame(colors: str) -> bool:
    s = 0
    w = 3
    countA = 0
    countB = 0
    while s <= len(colors) - 3:
        if colors[s:s + w] == 'AAA':
            countA += 1
        elif colors[s:s + w] == 'BBB':
            countB += 1
        s += 1
    if countA > countB:
        return True
    else:
        return False


winnerOfGame("AAABABB")