import heapq
from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt

State = namedtuple("State", ["f", "room", "g", "h"])

def heuristic(room, adj_list):
    return min(adj_list[room], default=0)

def k_shortest_paths(N, K, adj_list):
    G = nx.DiGraph()  # 创建一个用于可视化的有向图
    edge_labels = {}  # 创建一个用于存储边标签的字典
    for key in adj_list:
        for val, weight in adj_list[key]:
            G.add_edge(key, val, weight=weight)
            edge_labels[(key, val)] = weight  # 保存边的长度

    start_room = 1
    goal_room = N
    visited = set()
    paths = []

    f = heuristic(start_room, adj_list)[1]
    initial_state = State(f, start_room, 0, heuristic(start_room, adj_list))
    open_set = [initial_state]
    heapq.heapify(open_set)

    while open_set and len(paths) < K:
        current_state = heapq.heappop(open_set)
        if current_state.room == goal_room:
            paths.append(current_state.g)
            continue
        visited.add(current_state)
        for next_room, cost in adj_list[current_state.room]:
            if State(f, next_room, current_state.g + cost, heuristic(next_room, adj_list)) not in visited:
                if isinstance(heuristic(next_room, adj_list), tuple):
                    f = current_state.g + cost + heuristic(next_room, adj_list)[1]
                else:
                    f = current_state.g + cost
                new_state = State(f, next_room, current_state.g + cost, heuristic(next_room, adj_list))
                heapq.heappush(open_set, new_state)
                
                # 可视化部分
                plt.figure(figsize=(10, 6))
                pos = nx.spring_layout(G)
                nx.draw_networkx_nodes(G, pos)
                nx.draw_networkx_labels(G, pos)
                nx.draw_networkx_edges(G, pos, edgelist=[(current_state.room, next_room)], width=2, edge_color='r', alpha=0.5)
                nx.draw_networkx_edges(G, pos, alpha=0.2)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)  # 在图上显示边的长度
                plt.show()

    if len(paths) < K:
        paths.extend([-1] * (K-len(paths)))
    return paths


if __name__ == "__main__":
    input_strs = ["5 6 4\n1 2 1\n1 3 1\n2 4 2\n2 5 2\n3 4 2\n3 5 2"]
    for input_str in input_strs:
        lines = input_str.strip().split("\n")
        N, M, K = map(int, lines[0].split())
        adj_list = {i: [] for i in range(1, N + 1)}

    for line in lines[1:]:
        x, y, d = map(int, line.split())
        if x < y:
            #x, y = y, x
            adj_list[x].append((y, d))

    paths = k_shortest_paths(N, K, adj_list)
    for path in paths:
        print(path)
    print("\n")
