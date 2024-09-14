def evalRPN(tokens) -> int:
    def fun(si, a, b):
        a = int(a)
        b = int(b)
        if si == '+':
            return a + b
        elif si == '-':
            return a - b
        elif si == '*':
            return a * b
        else:
            return int(a / b)

    while len(tokens) > 1:
        i = 0
        while i < len(tokens):
            if tokens[i] in ['+', '-', '*', '/']:
                now = fun(tokens[i], tokens[i - 2], tokens[i - 1])
                tokens[i - 2] = str(now)
                del tokens[i - 1:i + 1]
                break
            i += 1
    return tokens[0]


evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"])