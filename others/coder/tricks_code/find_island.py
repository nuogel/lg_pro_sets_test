import numpy.random

m = 10
n = 10

block = []

ma = numpy.random.randint(0, 2, (10, 10))


def check(i, j):
    if 0 <= i < m and 0 <= j < n and ma[i, j] == 1:
        return True
    else:
        return False


def search_4(i, j):
    found = 0
    if check(i, j):
        ma[i, j] = 0
        found = 1
        search_4(i + 1, j)
        search_4(i, j + 1)
        search_4(i, j - 1)
        search_4(i - 1, j)

    return found


count = 0
for i in range(m):
    for j in range(n):
        if search_4(i, j):
            count += 1
assert ma.sum()==0
print(count)
