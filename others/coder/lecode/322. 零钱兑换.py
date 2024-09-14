def coinChange(coins, amount: int) -> int:
    dp = [0]
    for amounti in range(1, amount + 1):
        mincoins = 10000
        for coini in coins:
            if amounti - coini >= 0 and dp[amounti - coini] != -1:
                mincoins = min(mincoins, dp[amounti - coini] + 1)
        if mincoins != 10000:
            dp.append(mincoins)
        else:
            dp.append(-1)

    return dp[-1]


coinChange([2, 5, 10, 1], 27)


def tryit(coins, amount):
    dp=[0]
    dp+=[10000]*(amount)
    for  ai in range(1, amount+1):
        for ci in coins:
            if ai-ci>=0 and dp[ai-ci]!=10000:
                dp[ai] = min(dp[ai], dp[ai-ci]+1)

    return dp[-1]

tryit([2, 5, 10, 1], 27)
