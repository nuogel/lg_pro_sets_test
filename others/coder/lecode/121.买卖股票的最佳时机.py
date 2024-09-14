def maxProfit(prices) -> int:
    maxprice = 0
    pid = []
    for i in range(len(prices)):
        if pid == [] or prices[i] > prices[pid[-1]]:
            pid.append(i)
            zero = prices[pid[0]]
            popi = prices[pid[-1]]
            maxprice = max(maxprice, popi - zero)
        else:
            while pid != [] and prices[i] < prices[pid[-1]]:
                pid.pop()
            pid.append(i)

    return maxprice

maxProfit([7,1,5,3,6,4])