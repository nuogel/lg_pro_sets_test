'''
01背包(ZeroOnePack)： 有N件物品和一个容量为V的背包，每种物品均只有一件。第i件物品的费用是c[i]，价值是w[i]。求解将哪些物品装入背包可使价值总和最大。
多重背包(MultiplePack)： 有N种物品和一个容量为V的背包，第i种物品最多有n[i]件可用。每件费用是c[i]，价值是w[i]。求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大。
'''


# 01背包(ZeroOnePack)
def ZeroOnePack(value, weight, W):
    value.insert(0, 0)
    weight.insert(0, 0)
    maxvalue = [[0 for i in range(W + 1)] for i in range(len(weight))]

    for i in range(1, len(weight)):
        for j in range(1, W + 1):
            if weight[i] <= j:
                maxvalue[i][j] = max(maxvalue[i - 1][j], maxvalue[i - 1][j - weight[i]] + value[i])
            else:
                maxvalue[i][j] = maxvalue[i - 1][j]

    return maxvalue[-1][-1]


ZeroOnePack(value=[1, 6, 18, 22, 28], weight=[1, 2, 5, 6, 7], W=11)

'''
完全背包(CompletePack)： 有N种物品和一个容量为V的背包，每种物品都有无限件可用。第i种物品的费用是c[i]，价值是w[i]。求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大。
'''


# 完全背包(CompletePack)
def CompletePack(value, weight, W):
    value.insert(0, 0)
    weight.insert(0, 0)
    maxvalue = [[0 for i in range(W + 1)] for i in range(len(weight))]

    for i in range(1, len(weight)):
        for j in range(1, W + 1):
            if weight[i] <= j:
                # 01 背包：maxvalue[i][j] = max(maxvalue[i - 1][j], maxvalue[i - 1][j - weight[i]] + value[i])
                maxvalue[i][j] = max(maxvalue[i - 1][j], maxvalue[i][j - weight[i]] + value[i])
            else:
                maxvalue[i][j] = maxvalue[i - 1][j]
    return maxvalue[-1][-1]


CompletePack(value=[1, 6, 18, 22, 28], weight=[1, 2, 5, 6, 7], W=11)


def tryit01(value, weight, W):
    value.insert(0, 0)
    weight.insert(0, 0)
    ll = len(value)
    maxvalue = [[0 for i in range(W + 1)] for j in range(ll)]

    for i in range(1, ll):
        for wi in range(1, W + 1):
            if weight[i] <= wi:
                maxvalue[i][wi] = max(maxvalue[i - 1][wi], maxvalue[i][wi - weight[i]] + value[i])
            else:
                maxvalue[i][wi] = maxvalue[i - 1][wi]
    return maxvalue[-1][-1]


tryit01(value=[1, 6, 18, 22, 28], weight=[1, 2, 5, 6, 7], W=11)
