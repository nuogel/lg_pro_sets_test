def fistchar(str):
    for s in str:
        if str.count(s) == 1:
            return str.index(s)
    return -1


if __name__ == '__main__':
    print(fistchar('google'))
