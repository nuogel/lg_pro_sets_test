def solve(s):
    output = []

    def trackback(path, s):
        if s=='':
            output.append(path[:])
            return

        for i in range(len(s)+1):
            if s[:i]=='' or s[:i]!=s[:i][::-1]:
                continue
            path.append(s[:i])
            trackback(path, s[i:])
            path.pop(-1)

    trackback([], s)

    print(output)


solve('aab')
