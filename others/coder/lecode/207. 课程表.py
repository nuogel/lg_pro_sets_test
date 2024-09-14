def canfinish(numCourses: int, prerequisites):
    ind = [0 for i in range(numCourses)]
    need = [[] for i in range(numCourses)]

    for cur, pre in prerequisites:
        ind[cur]+=1
        need[pre].append(cur)

    q =[]
    for i, indi in enumerate(ind):
        if not indi:
            q.append(i)
    res=[]
    while q:
        qi = q.pop(0)
        numCourses-=1
        res.append(qi)
        for j in need[qi]:
            ind[j]-=1
            if not ind[j]:
                q.append(j)
    return not numCourses


from collections import deque

def canFinish2(numCourses: int, prerequisites) -> bool:
    indegrees = [0 for _ in range(numCourses)]
    adjacency = [[] for _ in range(numCourses)]
    queue = deque()
    # Get the indegree and adjacency of every course.
    for cur, pre in prerequisites:
        indegrees[cur] += 1
        adjacency[pre].append(cur)
    # Get all the courses with the indegree of 0.
    for i in range(len(indegrees)):
        if not indegrees[i]: queue.append(i)
    # BFS TopSort.
    while queue:
        pre = queue.popleft()
        numCourses -= 1
        for cur in adjacency[pre]:
            indegrees[cur] -= 1
            if not indegrees[cur]: queue.append(cur)
    return not numCourses

def tryit(num, courses):
    indegree = [0 for i in range(num)]




canFinish2(2,[[1,0]])
canfinish(2,[[1,0]])


