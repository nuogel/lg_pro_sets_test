def sumlist(am, an):
    # n = an - am + 1
    sumout = (an - am + 1) * (am + an) / 2
    return sumout


def findan(n):
    if 100 % n == 0 and (n + 1) % 2 == 0:
        return True
    else:
        return False


def findlist(num):
    i = 1
    j = 2
    out = []
    while i < j < 100:
        if sumlist(i, j) < num:
            j += 1
        elif sumlist(i, j) > num:
            i += 1
        else:
            out.append([ii for ii in range(i, j+1)])
            i+=1
    return out




if __name__ == '__main__':
    # print(sumlist(4, 6))
    print(findlist(9))
