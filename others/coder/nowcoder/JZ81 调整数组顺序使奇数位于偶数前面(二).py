def order(array):
    out = []
    for arr in array:
        if arr % 2 == 0:
            out.append(arr)
        else:
            out.insert(0, arr)
    return out
