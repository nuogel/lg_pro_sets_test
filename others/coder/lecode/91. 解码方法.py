def numDecodings(s: str) -> int:
    if not s or s[0] == '0':
        return 0

    al = [str(i) for i in range(1,27)]
    ecode = [1]
    for i in range(1, len(s)):
        if s[i] not in al:
            if s[i-1:i+1] not in al:
                return 0
            if s[i-1:i+1] in al:
                ecode.append(ecode[-2])
        else:
            if s[i-1:i+1] not in al:
                ecode.append(ecode[-1])
            if s[i-1:i+1] in al:
                ecode.append(ecode[i-2]+ecode[-1])
    return ecode[-1]


numDecodings("1123")