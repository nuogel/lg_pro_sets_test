# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
#
# @param root TreeNode类
# @param target int整型
# @return int整型二维数组
#
import math
math.fsum([4])
a=sum([4])
class Solution:
    def FindPath(self, root: TreeNode, target: int) -> List[List[int]]:
        # write code here
        if not root or target == 0:
            return []
        q = [root]
        out = []
        numq = [[root.val]]
        finalq = []
        while q:
            qi = q.pop(0)
            ni = numq.pop(0)
            if qi.left:
                q.append(qi.left)
                nl = ni.copy()
                nl.append(qi.left.val)
                numq.append(nl)

            if qi.right:
                q.append(qi.right)
                nr = ni.copy()
                nr.append(qi.right.val)
                numq.append(nr)

            if not qi.left and not qi.right:
                finalq.append(ni)
        for num in finalq:
            if sum(num) == target:
                out.append(num)
        return out
