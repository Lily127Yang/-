import heapq # 用于实现优先队列的工具
from collections import namedtuple # 用于创建简单的对象类型

# 定义一个命名元组，表示搜索状态，其中f是预测的总代价，room表示当前房间，g是已知代价，h是启发式函数的值。
State = namedtuple("State", ["f", "room", "g", "h"])

# 启发式函数定义：返回一个给定房间到其相邻房间的最小代价。
def heuristic(room, adj_list):
    return min(adj_list[room], default=0)

# K最短路径算法函数
def k_shortest_paths(N, K, adj_list):
    start_room = 1 # 起始房间，固定为1
    goal_room = N # 目标房间，固定为N
    visited = set()  # 创建一个集合用于存储已经访问过的状态
    paths = [] # 用于存储找到的路径长度

    # 计算起始状态的代价
    f = heuristic(start_room, adj_list)[1]
    # 创建初始状态对象
    initial_state = State(f, start_room, 0, heuristic(start_room, adj_list))
    open_set = [initial_state] # 创建优先队列，初始为起始状态
    heapq.heapify(open_set)  # 转为堆结构

    # 当优先队列不为空且未找到足够的路径时，继续搜索
    while open_set and len(paths) < K:
        current_state = heapq.heappop(open_set) # 弹出代价最小的状态
        # 如果当前房间是目标房间，将其代价加入到路径列表
        if current_state.room == goal_room:
            paths.append(current_state.g)
            continue
        visited.add(current_state) # 将当前状态加入到已访问集合
         # 遍历当前房间的相邻房间
        for next_room, cost in adj_list[current_state.room]:
            # 如果这个相邻房间的状态未被访问过，将其加入优先队列
            if State(f, next_room, current_state.g + cost, heuristic(next_room, adj_list)) not in visited:
                 # 如果启发式函数返回一个元组，取其第二个值作为h
                if isinstance(heuristic(next_room, adj_list), tuple):
                    f = current_state.g + cost + heuristic(next_room, adj_list)[1]
                else:
                    f = current_state.g + cost
                new_state = State(f, next_room, current_state.g + cost, heuristic(next_room, adj_list))
                heapq.heappush(open_set, new_state)

    # 如果未找到足够的K个路径，将-1填充至paths
    if len(paths) < K:
        paths.extend([-1] * (K-len(paths)))
    return paths

if __name__ == "__main__":
    # 测试数据
    input_strs = ["5 6 4\n1 2 1\n1 3 1\n2 4 2\n2 5 2\n3 4 2\n3 5 2",
                "6 9 4\n1 2 1\n1 3 3\n2 4 2\n2 5 3\n3 6 1\n4 6 3\n5 6 3\n1 6 8\n2 6 4",
                "7 12 6\n1 2 1\n1 3 3\n2 4 2\n2 5 3\n3 6 1\n4 7 3\n5 7 1\n6 7 2\n1 7 10\n2 6 4\n3 4 2\n4 5 1",
                "5 8 7\n1 2 1\n1 3 3\n2 4 1\n2 5 3\n3 4 2\n3 5 2\n1 4 3\n1 5 4",
                "6 10 8\n1 2 1\n1 3 2\n2 4 2\n2 5 3\n3 6 3\n4 6 3\n5 6 1\n1 6 8\n2 6 5\n3 4 1"]
    # 遍历测试数据
    for input_str in input_strs:
        lines = input_str.strip().split("\n")
        N, M, K = map(int, lines[0].split())
        adj_list = {i: [] for i in range(1, N + 1)}

        # 填充邻接列表
        for line in lines[1:]:
            x, y, d = map(int, line.split())
            if x < y:
                #x, y = y, x
                adj_list[x].append((y, d))

        # 调用k_shortest_paths函数并打印结果
        paths = k_shortest_paths(N, K, adj_list)
        for path in paths:
            print(path)
        print("\n")
