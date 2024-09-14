'''
给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。

返回被除数 dividend 除以除数 divisor 得到的商。

整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2

示例 1:

输入: dividend = 10, divisor = 3
输出: 3
解释: 10/3 = truncate(3.33333..) = truncate(3) = 3

示例 2:

输入: dividend = 7, divisor = -3
输出: -2
解释: 7/-3 = truncate(-2.33333..) = -2
'''

def solve(a,b):
    sign = 1
    if a*b<0:
        sign=-1
    a = abs(a)
    b = abs(b)

    def div(a,b):
        if a<b:return 0
        b2 = b
        s=1
        while b2+b2<a:
            s*=2
            b2*=2
        return s+div(a-b*s, b)

    if b==1:
        out = sign*a
    else:
        out = sign*div(a, b)
    return max(min(2**31-1, out), -2**32)


solve(100,3)
