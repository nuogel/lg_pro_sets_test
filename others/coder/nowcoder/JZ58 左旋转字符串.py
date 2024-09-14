def leftmove(str, n):
    if len(str)<1:
        return str
    n = n % len(str)
    left = str[:n]
    right = str[n:]
    out = right + left
    return out


if __name__ == '__main__':
    str = '12345678'
    n=2
    print(leftmove(str, n))
