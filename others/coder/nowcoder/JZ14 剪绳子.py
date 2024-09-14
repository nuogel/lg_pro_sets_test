class Solution:
    def cutRope0(self, n: int) -> int:  # 动态规划
        if n < 2:
            return 0
        if n == 3:
            return 2
        f = [-1] * (n + 1)
        for i in range(5):
            f[i] = i
        for j in range(5, n + 1):
            for i in range(1, j):
                f[j] = max(i * f[j - i], f[j])
        return f[n]

    def cutRopex3(self, n: int) -> int:  # 由于答案过大，请对 998244353 取模。
        # 除3法：如果我们把长度为n的绳子分为x段，则每段只有在长度相等的时候乘积最大，那么每段的长度是n/x。
        # 所以他们的乘积是(n/x)^x。我们来对这个函数求导得到e=2.7最大；取近似3.
        mod = 998244353
        if n < 2:
            return 0
        if n == 2:
            return 1
        if n == 3:
            return 2
        if n == 4:
            return 4

        def fastpow(a, b):  # 递归求power % mod
            # out=pow(a,b)
            if b == 0:
                return 1
            if b == 1:
                return a
            b2 = b // 2
            out = fastpow(a, b2)
            if b % 2 == 1:
                out = out * out * a
            else:
                out = out * out
            return out % mod

        if n % 3 == 0:
            out = fastpow(3, n // 3)
        elif n % 3 == 1:
            out = fastpow(3, n // 3 - 1) * 4
        else:
            out = fastpow(3, n // 3) * 2
        return out % mod

        '''
        下面的时间复杂度过大
        '''
        # if n < 2:
        #     return 0
        # if n == 2:
        #     return 1
        # if n == 3:
        #     return 2
        # if n == 4:
        #     return 4
        # res = 1
        # while n > 4:
        #     n -= 3
        #     res *= 3
        #     res %= mod
        # return res * n % mod

    def cutRope(self, n: int) -> int:  # 记忆递归
        if n < 2:
            return 0
        if n == 3:
            return 2
        self.mem = [-1] * n
        return self.maxrope(n)

    def maxrope(self, n):
        if n <= 4:
            return n
        if self.mem[n - 1] != -1:
            return self.mem[n - 1]
        maxr = 0
        for i in range(1, n):
            maxr = max(maxr, i * self.maxrope(n - i))
        self.mem[n - 1] = maxr
        return maxr


if __name__ == '__main__':
    s = Solution()
    # print(s.cutRope(5))
    # print(s.cutRope0(5))
    print(s.cutRopex3(874520))
