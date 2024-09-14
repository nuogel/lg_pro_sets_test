#  字符串数字相加

def add(str1, str2):
    int1 = [int(i) for i in str1]
    int2 = [int(i) for i in str2]

    l1 = len(int1)
    l2 = len(int2)

    if l1 > l2:
        for i in range(l1 - l2):
            int2.insert(0, 0)
    else:
        for i in range(l2 - l1):
            int1.insert(0, 0)
    out = []
    a = 0
    while int1:
        plus = int1.pop(-1) + int2.pop(-1)
        plus += a
        if plus >= 10:
            a = 1
            b = plus - 10
        else:
            a = 0
            b = plus
        out.insert(0, str(b))
    strout = ''.join(out)
    return strout


add('1234', '567')
