def largestRectangleArea(heights) -> int:
    area = 0
    for i in range(len(heights)):
        s = i
        e = i
        while s <= e < len(heights):
            if heights[e] == 0:
                s = e
                e = s + 1
            sarea = (e - s+1) * min(heights[s:e + 1])
            if sarea > area:
                area = sarea
            e += 1
    return area


def largestRectangleArea2(heights) -> int:
    area=0
    for i in range(len(heights)):
        hi=heights[i]
        left=i
        right=i
        while left>=0 and heights[left]>=hi:
                left-=1

        while right<len(heights) and heights[right]>=hi:
                right+=1

        ai=(right-left-1)*hi
        if ai>area:
            area=ai
    return area

def largestRectangleArea3(heights) -> int:
    starck = [0]
    heights.insert(0, 0)
    heights.append(0)
    id = 0
    areamax = 0
    while id < len(heights):
        while id < len(heights) and heights[starck[-1]] <= heights[id]:
            starck.append(id)
            id += 1

        while id < len(heights) and heights[starck[-1]] > heights[id]:
            ipop = starck.pop()
            right=id-1
            left=starck[-1]+1
            areamax = max(areamax, (right-left+1) * heights[ipop])
    return areamax


largestRectangleArea3([2,1,2])

largestRectangleArea([2,1,5,6,2,3])
