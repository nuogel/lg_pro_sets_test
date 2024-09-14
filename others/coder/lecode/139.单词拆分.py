def wordBreak(s: str, wordDict) -> bool:
    # aldict = {}
    # for wi in wordDict:
    #     if wi[0] in aldict:
    #         aldict[wi[0]].append(wi)
    #     else:
    #         aldict[wi[0]] = [wi]
    # output = False
    # mem=['']
    # def solve(s, output):
    #     if not s:
    #         output = True
    #         return output
    #     s0 = s[0]
    #     if s in mem:
    #         return True
    #     if s0 not in aldict:
    #         return False
    #     for wi in aldict[s0]:
    #         memlast=mem[-1]
    #         if wi == s[:len(wi)]:
    #             mem.append(memlast+wi)
    #             sout = s[len(wi):]
    #             output = output or solve(sout, output)
    #             if output:
    #                 return output
    #     return output
    # out = solve(s, output)
    # return out

    output = [0]
    for i in range(len(s)):
        ll = len(output)
        for oi in range(ll):
            si = s[output[oi]:i+1]
            if si in wordDict:
                output.append(i+1)
                break
    if output[-1] == len(s):
        return True
    else:
        return False


wordBreak("leetcode",
["leet","code"]


)