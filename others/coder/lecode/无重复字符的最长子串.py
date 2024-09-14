def lengthOfLongestSubstring(s: str) -> int:
        i=0
        j=0
        _max=0
        while i<=j<len(s):
            if s[j] not in s[i:j]:
                j+=1
                _max=max(len(s[i:j]),_max)
            else:
                i+=s[i:j].index(s[j])+1
        return _max


lengthOfLongestSubstring(" ")