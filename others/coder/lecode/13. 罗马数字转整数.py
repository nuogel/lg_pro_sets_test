s='IV'

roma = ['I', 'IV', 'V', 'IX', 'X', 'XL', 'L', 'XC','C', 'CD','D', 'CM', 'M']
romai = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]

num=0
while s:
    if s[:2] in roma:
        num+=romai[roma.index(s[:2])]
        s=s[2:]
    else:
        num += romai[roma.index(s[:1])]
        s=s[1:]
print(num)

