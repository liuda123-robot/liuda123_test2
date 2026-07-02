import numpy as np
import heapq#模拟最小堆，永远将值最小的元素排在最前面
import math

class Node:
    def __init__(self, level, path, bound, current_cost):
        """
        搜索树中的一个节点，代表一条探索到一半的路线
        :param level: 当前所在的树深度（已经走了几个城市）
        :param path: 当前走过的路线列表，如 [0, 1, 3]
        :param bound: 当前这半条路线的理论下界 (Lower Bound)
        :param current_cost: 当前这半条路线的实际真实花费
        """
        self.level = level
        self.path = path
        self.bound = bound
        self.current_cost = current_cost

    # 定义比较方法，为了让 heapq 优先队列根据 bound 从小到大排序 (Best-First Search)
    def __lt__(self, other):
        return self.bound < other.bound

def get_min_edges(dist_matrix):
    """
    预计算：计算每个城市出发的最短边和次短边。
    用于快速计算下界。
    """
    n = len(dist_matrix)
    min_edges = []
    for i in range(n):
        # 排除对角线的 0
        edges = [dist_matrix[i][j] for j in range(n) if i != j]
        edges.sort()
        # 存入最短和次短的边
        min_edges.append((edges[0],edges[1]))
    return min_edges

def tsp_branch_and_bound(dist_matrix):
    n = len(dist_matrix)
    # 获取每个城市的最短出边，用于估算下界
    min_edges = get_min_edges(dist_matrix)
    
    # 优先队列（存活的树分支）
    pq = []
    
    # 记录当前的全局最优解（上界 Upper Bound）
    min_path_cost = math.inf
    best_path = []

    # 1. 计算根节点的初始下界
    # 最简单的有效下界：每个城市走最便宜的边的总和
    initial_bound = sum([min_edges[i][0] for i in range(n)])
    
    # 创建根节点（从城市0出发）
    root = Node(level=1, path=[0], bound=initial_bound, current_cost=0)
    heapq.heappush(pq, root)

    nodes_explored = 0 # 统计探索的节点数（用于观察剪枝威力）

    # 2. 开始探索搜索树
    while pq:
        # 取出当前下界最小、最有潜力的节点
        current_node = heapq.heappop(pq)
        nodes_explored += 1

        # 【核心剪枝逻辑】
        # 如果这个节点哪怕发挥到极限(bound)，也比我们手头的最优解(min_path_cost)差，直接扔掉！
        if current_node.bound >= min_path_cost:
            continue 

        # 如果这是一条完整路径的最后一步（所有城市都走完了，准备回起点）
        if current_node.level == n:
            last_city = current_node.path[-1]
            first_city = current_node.path[0]
            # 加上回家的路费
            total_cost = current_node.current_cost + dist_matrix[last_city][first_city]
            
            # 如果比当前擂主厉害，更新擂主（收紧上界）
            if total_cost < min_path_cost:
                min_path_cost = total_cost
                best_path = current_node.path + [first_city]
            continue

        # 如果没走完，继续向下分支（尝试去下一个没去过的城市）
        current_city = current_node.path[-1]
        
        for next_city in range(n):
            if next_city not in current_node.path:
                # 计算走这条路的真实花费
                next_cost = current_node.current_cost + dist_matrix[current_city][next_city]
                
                # 计算新节点的下界估算值 (非常简化的估算策略)
                # 新下界 = 原下界 - 当前城市原本估算的最小出边 + 真实走的出边
                # 这里为了保证绝对安全(绝不估高)，我们直接用剩余未访问城市的最小出边来估算
                remaining_estimate = 0
                for v in range(n):
                    if v not in current_node.path and v != next_city:
                         remaining_estimate += min_edges[v][0]
                
                # 下界 = 当前真实花费 + 下一个城市回家的最小代价 + 其他还没去城市的最小出站代价
                next_bound = next_cost + min_edges[next_city][0] + remaining_estimate

                # 【再次剪枝判断】只有估算下界小于当前擂主成绩，才有资格加入队列
                if next_bound < min_path_cost:
                    next_node = Node(
                        level=current_node.level + 1,
                        path=current_node.path + [next_city],
                        bound=next_bound,
                        current_cost=next_cost
                    )
                    heapq.heappush(pq, next_node)

    return min_path_cost, best_path, nodes_explored

# ================= 测试用例 =================
if __name__ == "__main__":
    # 使用和上次 DP 完全相同的 4 城市矩阵
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])

    print("距离矩阵:\n", distance_matrix)
    best_cost, best_route, explored = tsp_branch_and_bound(distance_matrix)
    
    print("\n分支限界法求解结果:")
    print(f"最短路径距离: {best_cost}")
    print("最优行驶路线: " + " -> ".join(map(str, best_route)))
    print(f"总共探索的节点数: {explored} (如果穷举，4个城市有极多状态，B&B通常会少很多)")