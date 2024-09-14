# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.unique = []
        self.uniqueremoved=[]

    def FirstAppearingOnce(self):
        if self.ch not in self.unique and self.ch not in self.uniqueremoved:
            self.unique.append(self.ch)
        else:
            if self.ch in self.unique:
                self.unique.remove(self.ch)
            self.uniqueremoved.append(self.ch)
        if self.unique==[]:
            return '#'
        else:
            return self.unique[0]


    # write code here
    def Insert(self, char):
        # write code here
        self.ch = char
        out = self.FirstAppearingOnce()
        return out


if __name__ == '__main__':
    char = 'googgle'
    s = Solution()
    out = ''
    for ch in char:
        out += s.Insert(ch)
    print(out)
