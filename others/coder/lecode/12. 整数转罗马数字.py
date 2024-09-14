def intToRoman(num: int) -> str:
    roma = ['I', 'IV', 'V', 'IX', 'X', 'XL', 'L', 'XC', 'C', 'CD', 'D', 'CM', 'M'][::-1]
    romai = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000][::-1]

    romanum = ''
    for i in range(len(roma)):
        while num > romai[i]:
            romanum += roma[i]
            num -= romai[i]
    return romanum


intToRoman(3999)
