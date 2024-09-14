
def titleToNumber(columnTitle: str) -> int:
    def alphanum(s):
        return ord(s) - 64

    out = 0
    ll = len(columnTitle)
    for i in range(ll):
        out+=alphanum(columnTitle[i])*26**(ll-1-i)
    return out


titleToNumber('ZY')
