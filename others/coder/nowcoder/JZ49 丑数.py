def fun(n):  # 质因数分解
    out = []
    while n != 1:
        flag = False
        for i in range(2, n):
            if n % i == 0:
                out.append(i)
                n = n // i
                flag = True
                break
        if not flag:
            out.append(n)
            return out


def uglydata(n):  # 暴力求解
    while n % 2 == 0:
        n /= 2
    while n % 3 == 0:
        n /= 3
    while n % 5 == 0:
        n /= 5
    if n == 1:
        return True
    else:
        return False


def uglydataindex(index):
    outdata = [1]
    i = 1
    while 1:
        i += 1
        if len(outdata) == index:
            return outdata[-1]
        else:
            if uglydata(i):
                outdata.append(i)


def uglydataindex_2(index):
    out = [1]
    while len(out) != index:
        l = len(out)
        for i in range(l):
            out += [out[i] * 2, out[i] * 3, out[i] * 5]
            out = list(set(out))
            out.sort()
        l += 1
        out = out[: l]
    return out[index - 1]


def official(idx):
    if idx < 1:  # 非法输入的情况  ugly = 2x3y5z
        return 0
    lst = [1]
    i = 0  # i为指向下一个*2可能成为下一个丑数的数的位置的指针
    j = 0  # j为指向下一个*3可能成为下一个丑数的数的位置的指针
    k = 0  # k为指向下一个*5可能成为下一个丑数的数的位置的指针
    while len(lst) < idx:  # 当得到第idx个丑数的时候，循环停止
        now = min(lst[i] * 2, lst[j] * 3, lst[k] * 5)  # 三个指针运算的结果中找，下一个丑数
        lst.append(now)  # 将下一个丑数入队
        if now == lst[i] * 2:  # 下一个丑数可以由v[i]*2得到，则i指针后移
            i += 1
        if now == lst[j] * 3:  # 下一个丑数可以由v[j]*3得到，则j指针后移
            j += 1
        if now == lst[k] * 5:  # 下一个丑数可以由v[k]*5得到，则k指针后移
            k += 1
    return lst[idx - 1]  # 返回答案,如果idx==1，now没有定义，依旧会CE，所以此处不能写now


if __name__ == '__main__':
    print(official(1500))
    print(uglydataindex_2(1500))
