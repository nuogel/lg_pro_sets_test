def isPalindrome(s: str) -> bool:
    ll = len(s)

    slist = []
    for i in range(ll):
        if s[i].isalnum():
            slist.append(s[i].lower())
    return slist == slist[::-1]


print(isPalindrome("0P"))