'''
1.     1
2.     11
3.     21
4.     1211
5.     111221
第一项是数字 1
描述前一项，这个数是 1 即 “ 一 个 1 ”，记作 "11"
描述前一项，这个数是 11 即 “ 二 个 1 ” ，记作 "21"
描述前一项，这个数是 21 即 “ 一 个 2 + 一 个 1 ” ，记作 "1211"
描述前一项，这个数是 1211 即 “ 一 个 1 + 一 个 2 + 二 个 1 ” ，记作 "111221"
'''


def countAndSay(n: int) -> str:
    def fun1(num):
        mark = ''
        i = 0
        while i < len(num):
            count = 1
            si = num[i]
            while i + 1 < len(num) and num[i] == num[i + 1]:
                count += 1
                i += 1

            mark += str(count) + si
            i += 1
        return mark

    strm = '1'
    for i in range(1, n):
        strm = fun1(strm)

    return strm

countAndSay(4)