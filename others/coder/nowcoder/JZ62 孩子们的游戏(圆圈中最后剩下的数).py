def lastremain(n, m):
    if n < 1:
        return 0
    _list = list(range(n))
    while len(_list) > 1:
        _m = m % len(_list)
        del _list[_m-1]
        if _m>0:
            _list = _list[_m-1:]+_list[:_m-1]
    return _list[0]


if __name__ == '__main__':
    print(lastremain(10, 17))
