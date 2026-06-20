import numpy as np
from functools import lru_cache

def tsp_dynamic_programming(dist_matrix):
    """
    使用状态压缩动态规划求解 TSP 问题
    :param dist_matrix: 城市之间的距离矩阵 (2D NumPy Array)
    :return: (最短总距离, 最优路径列表)
    """
    n = len(dist_matrix)
    # 目标状态：所有城市都已访问（n 个 1）
    ALL_VISITED = (1 << n) - 1

    # 使用 lru_cache 自动进行状态缓存 (Memoization)
    # 状态由两部分组成：当前已访问的城市集合(mask)，以及当前所在的城市(pos)
    @lru_cache(maxsize=None)
    def dp(mask, pos):
        # 边界条件：如果所有城市都访问过了，返回当前城市回到起点(城市0)的距离
        if mask == ALL_VISITED:
            return dist_matrix[pos][0], [0]

        min_dist = float('inf')
        best_path = []

        # 遍历所有可能的下一个城市
        for next_city in range(n):
            # 位运算：检查 next_city 是否已经被访问过
            # (1 << next_city) 是将 1 左移 next_city 位，与 mask 做按位与
            if (mask & (1 << next_city)) == 0:
                # 如果没访问过，将其加入已访问集合（按位或）
                new_mask = mask | (1 << next_city)
                
                # 递归计算后续的距离和路径
                next_dist, next_path = dp(new_mask, next_city)
                
                # 当前的总距离 = 到下一个城市的距离 + 后续的最短距离
                total_dist = dist_matrix[pos][next_city] + next_dist

                # 寻找最小距离
                if total_dist < min_dist:
                    min_dist = total_dist
                    best_path = [next_city] + next_path

        return min_dist, best_path

    # 从城市 0 出发，初始状态 mask = 1 (即 00...01)
    # 因为起点已经确定访问过，所以传 (1, 0)
    shortest_dist, optimal_path = dp(1, 0)
    
    # 完整路径需要加上起点
    full_path = [0] + optimal_path
    return shortest_dist, full_path

# ================= 测试用例 =================
if __name__ == "__main__":
    # 构建一个 4 个城市的距离矩阵 (对称矩阵)
    # 假设对角线（自己到自己）距离为 0
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])

    print("距离矩阵:\n", distance_matrix)
    dist, path = tsp_dynamic_programming(distance_matrix)
    
    print("\nDP 求解结果:")
    print(f"最短路径距离: {dist}")
    # 将节点编号转换为更直观的箭头表示
    print("最优行驶路线: " + " -> ".join(map(str, path)))