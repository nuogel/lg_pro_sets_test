def myAtoi(s: str) -> int:
    s = s.strip()

    sign = 1
    if s[0] == '-':
        sign = -1
        s = s[1:]
    sint = ''
    while s:
        if '0' <= s[0] <= '9':
            sint += s[0]
            s = s[1:]
        else:
            break
    if sint:
        sint = int(sint) * sign
        if -2 ** 31 <= sint <= 2 ** 31 - 1:
            return sint
    return 0

myAtoi("-91283472332")
