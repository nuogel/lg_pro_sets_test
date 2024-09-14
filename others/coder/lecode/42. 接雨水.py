def trap( height) -> int:
    dropleft = []
    dropright = []
    maxleft = 0
    maxright = 0
    ll = len(height)
    for i in range(ll):
        maxleft = max(maxleft, height[i])
        maxright = max(maxright, height[ll - 1 - i])
        dropleft.append(maxleft)
        dropright.insert(0,maxright)

    drops = 0
    for i in range(ll):
        drops += min(dropleft[i], dropright[i]) - height[i]
    return drops

height = [0,1,0,2,1,0,1,3,2,1,2,1]
trap(height)