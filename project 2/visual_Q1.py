import heapq
from collections import namedtuple
import tkinter as tk
from time import sleep

# 使用namedtuple定义一个名为State的数据结构，它包含以下五个属性：
# f: 评估函数值，用于估计从当前状态到目标状态的代价
# board: 当前棋盘状态
# zero_idx: 0(空块)在棋盘上的位置索引
# g: 从初始状态到当前状态的实际代价
# h: 从当前状态到目标状态的估计代价

State = namedtuple("State", ["f", "board", "zero_idx", "g", "h", "prev"])

# 定义曼哈顿距离函数，计算当前状态到目标状态的距离
def manhattan_distance(board):
    distance = 0
    for i in range(9): # 对于棋盘上的每一个位置
        if board[i] != 0: # 0是空块，不需要计算距离
            # 根据题目的目标状态计算每个块应该在的位置
            correct_pos = (board[i] - 1) % 8
            # 计算当前块和应该在的位置的距离，并累加
            distance += abs(i // 3 - correct_pos // 3) + abs(i % 3 - correct_pos % 3)
    return distance

# 定义主解决函数
def solve(initial_board):
    # 目标状态
    goal_board = (1, 3, 5, 7, 0, 2, 6, 8, 4)
    # 评估初始状态到目标状态的代价
    f = manhattan_distance(initial_board)
    # 创建初始状态
    initial_state = State(f, initial_board, initial_board.index(0), 0, manhattan_distance(initial_board), None)
# 使用一个列表作为优先队列存放待扩展的状态
    open_set = [initial_state]
    heapq.heapify(open_set)# 转化为最小堆结构，使得代价最小的状态能够被优先扩展

    visited = set()# 用于记录已经访问过的状态

    while open_set:  # 当待扩展的状态列表不为空时
        current_state = heapq.heappop(open_set) # 取出代价最小的状态

 # 如果当前状态为目标状态，返回到达该状态所需的步数
        #if current_state.board == goal_board:
            #return current_state.g
        if current_state.board == goal_board:
            return current_state
        visited.add(current_state.board) # 标记当前状态为已访问

       # 获取空块的位置
        x, y = current_state.zero_idx // 3, current_state.zero_idx % 3
         # 定义四个可能的移动方向
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            # 判断移动方向是否有效（是否超出棋盘边界）
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_zero_idx = new_x * 3 + new_y
                # 获取新的棋盘状态
                new_board = list(current_state.board)
                new_board[current_state.zero_idx], new_board[new_zero_idx] = new_board[new_zero_idx], new_board[current_state.zero_idx]
                new_board = tuple(new_board)
                # 如果新的状态没有被访问过
                if new_board not in visited:
                    # 计算新状态的评估代价
                    f = current_state.g + 1 + manhattan_distance(new_board)
                    new_state = State(f, new_board, new_zero_idx, current_state.g + 1, manhattan_distance(new_board), current_state)

                    # 创建新的状态并加入到待扩展的状态列表中
                    heapq.heappush(open_set, new_state)

def print_solution(state):
    path = []
    while state:
        path.append(state.board)
        state = state.prev
    for step, board in enumerate(reversed(path)):
        print(f"Step {step}:")
        for i in range(0, 9, 3):
            print(board[i:i+3])
        print()



def visualize_solution(state):
    path = []
    while state:
        path.append(state.board)
        state = state.prev

    root = tk.Tk()
    root.title("8-Puzzle Solver")

    # 设置主窗口的大小
    root.geometry("200x200")

    # 创建9个标签来表示棋盘的每一个块
    labels = [tk.Label(root, font=("Arial", 24), width=2, height=1) for _ in range(9)]

    # 根据棋盘状态更新标签
    def update_labels(board):
        for i, num in enumerate(board):
            labels[i].config(text=str(num) if num != 0 else "")
            labels[i].grid(row=i//3, column=i%3, padx=5, pady=5)

    # 逐步显示解决方案
    for step, board in enumerate(reversed(path)):
        update_labels(board)
        root.update()
        sleep(4)  # 等待1秒，然后显示下一步

    root.mainloop()

if __name__ == "__main__":
    # 定义几个测试的初始状态
    input_str = ["135720684"
                 #, 
                 # "105732684"
                 # , "015732684"
                 # , "135782604"
                 # , "715032684"
                 ]
    for i in range(len(input_str)):
        # 将字符串转换为元组格式
        initial_board = tuple(map(int, input_str[i]))
         # 输出从初始状态到目标状态所需的步数
        final_state = solve(initial_board)
        if final_state:
            print_solution(final_state)
    for i in range(len(input_str)):
        initial_board = tuple(map(int, input_str[i]))
        final_state = solve(initial_board)
        if final_state:
            visualize_solution(final_state)