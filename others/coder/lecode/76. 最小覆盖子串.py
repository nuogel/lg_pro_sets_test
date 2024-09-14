def minWindow(s: str, t: str) -> str:
    tdict = {}
    sdict = {}
    for ti in t:
        if ti not in tdict:
            tdict[ti] = 1
            sdict[ti] = 0
        else:
            tdict[ti] += 1

    l = 0
    r = 0

    def checkwin(sd, td):
        for k, v in td.items():
            if sd[k] - td[k] < 0:
                return False
        return True

    minl = len(s) + 10
    outstr = ''
    swin = s[l:r]
    while l <= r and r <= len(s):
        while checkwin(sdict, tdict.copy()):
            if r - l < minl:
                outstr = s[l:r]
                minl = r - l
            l += 1
            if s[l - 1] in sdict:
                sdict[s[l - 1]] -= 1
        r += 1
        if r <= len(s) and s[r - 1] in tdict:
            sdict[s[r - 1]] += 1

    return outstr


import collections


def minWindow2(s: str, t: str) -> str:
    need = collections.defaultdict(int)
    for c in t:
        need[c] += 1
    needCnt = len(t)
    i = 0
    res = (0, float('inf'))
    for j, c in enumerate(s):
        if need[c] > 0:
            needCnt -= 1
        need[c] -= 1
        if needCnt == 0:  # 步骤一：滑动窗口包含了所有T元素
            while True:  # 步骤二：增加i，排除多余元素
                c = s[i]
                if need[c] == 0:
                    break
                need[c] += 1
                i += 1
            if j - i < res[1] - res[0]:  # 记录结果
                res = (i, j)
            need[s[i]] += 1  # 步骤三：i增加一个位置，寻找新的满足条件滑动窗口
            needCnt += 1
            i += 1
    return '' if res[1] > len(s) else s[res[0]:res[1] + 1]  # 如果res始终没被更新过，代表无满足条件的结果


minWindow2("ADOBECODEBANC",
           "ABC")
