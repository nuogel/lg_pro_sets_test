def divide(dividend: int, divisor: int) -> int:
    if divisor == 1: return max(min(2 ** 31 - 1, dividend), -2 ** 32)
    if divisor == -1: return max(min(2 ** 31 - 1, -dividend), -2 ** 32)

    if dividend * divisor > 0:
        sign = 1
    else:
        sign = -1
    out = 0
    dividend = abs(dividend)
    divisor = abs(divisor)

    def div(a, b):
        if a < b: return 0
        count = 1
        bb = b
        while bb+bb < a:
            count = count + count
            bb = bb + bb
        return count + div(a-bb, b)

    out = div(dividend, divisor)

    return out * sign


divide(10, 3)