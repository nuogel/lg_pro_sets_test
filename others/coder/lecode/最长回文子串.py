def longestPalindrome(s: str) -> str:
    maxl = 0
    strout = ''
    for i in range(len(s)):
        left = i - 1
        right = i + 1
        lenl = 1
        while left >= 0 and s[left] == s[i]:
            left -= 1
            lenl += 1
        while right < len(s) and s[right] == s[i]:
            right += 1
            lenl += 1
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
            lenl += 2
        if lenl >= maxl:
            maxl = lenl
            strout = s[left+1:right]

    return strout


longestPalindrome("eabcb")
