def power(x, n):
    def power(x, n):
        if n == 0:
            return 1
        if n % 2 == 1:
            ad = x
        else:
            ad = 1
        powr = power(x, n // 2)
        out = powr * powr * ad
        return out

    if n < 0:
        return 1 / power(x, -n)
    else:
        return power(x, n)


a = power(2, -2)
a = 0


def tryit(x, n):
    def powerit(x, n):

        if n == 0:
            return 1

        pow = powerit(x, n // 2)
        if n % 2 == 1:
            return pow * pow * x
        else:
            return pow * pow

    if n < 0:
        sign = -1
        n = -n
    else:
        sign = 1

    out = powerit(x, n)
    return out if sign else 1 / out

tryit(2, 10)
