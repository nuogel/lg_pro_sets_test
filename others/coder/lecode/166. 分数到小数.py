def solve(numerator: int, denominator: int):
    if numerator*denominator<0:
        sign = '-'
        numerator = abs(numerator)
        denominator = abs(denominator)
    else:
        sign = ''
    da = []
    db = []
    b = numerator

    while b != 0:
        a = b  // denominator
        da.append(a)
        b = b  % denominator
        if b in db:
            loopi = db.index(b)+1
            return sign+ str(da[0]) + '.' + (''.join(str(i) for i in da[1:loopi]) if da[1:loopi] else '') + '(' + ''.join(str(i) for i in da[loopi:]) + ')'

        db.append(b)
        b = b*10
    o2 = ''
    if da[1:]:
        o2 = '.' + ''.join(str(i) for i in da[1:])
    return sign+str(da[0]) + o2 if da else '0'


print(solve(7, -12))
