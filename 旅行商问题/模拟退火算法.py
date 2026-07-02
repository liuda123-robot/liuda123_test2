import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ================= 1. 基础工具函数 =================

def calculate_total_distance(route, dist_matrix):
    """计算一条完整路径的总距离"""
    dist = 0
    n = len(route)
    for i in range(n):
        # 依次连接当前城市与下一个城市，最后一个城市连回起点
        dist += dist_matrix[route[i]][route[(i + 1) % n]]
    return dist

def get_neighbor(route):
    """
    邻域动作：随机交换路径中的两个城市来产生新路径
    """
    new_route = route.copy()
    n = len(new_route)
    # 随机选两个不同的索引
    i, j = random.sample(range(n), 2)
    # 交换
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# ================= 2. 模拟退火核心算法 =================

def simulated_annealing(dist_matrix, initial_temp=1000, cooling_rate=0.99, min_temp=1, iter_per_temp=1000):
    """
    :param initial_temp: 初始高温 (醉酒极深)
    :param cooling_rate: 冷却率 α (醒酒速度)
    :param min_temp: 终止温度 (完全清醒)
    :param iter_per_temp: 每个温度下尝试探索的次数 (内循环)
    """
    n = len(dist_matrix)
    
    # 1. 初始解：随便给一个顺序 (如 0, 1, 2... n-1)
    current_route = list(range(n))
    current_cost = calculate_total_distance(current_route, dist_matrix)
    
    # 记录全局最优 (因为有时候算法最后一步可能跳出去了，我们要把历史最好的存下来)
    best_route = current_route.copy()
    best_cost = current_cost
    
    # 用于记录画图的收敛数据
    history_best_cost = []
    
    T = initial_temp
    
    # 外循环：控制温度下降
    while T > min_temp:
        # 内循环：在当前温度下，进行多次随机探索
        for _ in range(iter_per_temp):
            # 产生新路线 (邻域动作)
            new_route = get_neighbor(current_route)
            new_cost = calculate_total_distance(new_route, dist_matrix)
            
            # 能量差 (代价差)
            delta_e = new_cost - current_cost
            
            # Metropolis 准则
            if delta_e < 0:
                # 变好了，100% 接受
                current_route = new_route
                current_cost = new_cost
                # 更新全局最优
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_route = current_route.copy()
            else:
                # 变差了，以一定概率接受 (核心魔法！)
                # math.exp() 计算 e 的 x 次方
                probability = math.exp(-delta_e / T)
                if random.random() < probability: # random.random() 生成 0~1 的随机数
                    current_route = new_route
                    current_cost = new_cost
                    
        history_best_cost.append(best_cost)
        
        # 降温 (退火)
        T *= cooling_rate
        
    return best_route, best_cost, history_best_cost

# ================= 3. 测试与可视化 =================
if __name__ == "__main__":
    # 随机生成 30 个城市的 2D 坐标 (0~100)
    np.random.seed(42) # 固定随机种子，方便复现
    num_cities = 30
    cities_coords = np.random.rand(num_cities, 2) * 100
    
    # 计算距离矩阵 (欧氏距离)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(cities_coords[i] - cities_coords[j])
            
    print(f"开始求解 {num_cities} 个城市的 TSP...")
    
    # 运行模拟退火算法
    best_route, best_cost, history = simulated_annealing(
        dist_matrix, 
        initial_temp=1000, 
        cooling_rate=0.99, # 调整衰减率，0.95 降温较快，0.99 较慢但搜索更精细
        min_temp=0.1, 
        iter_per_temp=1000
    )
    
    print(f"最终最短路线距离: {best_cost:.2f}")
    
    # ======= 画图 =======
    plt.figure(figsize=(12, 5))
    
    # 图 1：收敛曲线
    plt.subplot(1, 2, 1)
    plt.plot(history, color='b', linewidth=2)
    plt.title('Convergence Curve (Cost vs Iterations)')
    plt.xlabel('Temperature Iterations')
    plt.ylabel('Best Distance')
    plt.grid(True)
    
    # 图 2：最终路径轨迹
    plt.subplot(1, 2, 2)
    # 把起点加到路线末尾，形成闭环
    plot_route = best_route + [best_route[0]]
    x_coords = [cities_coords[i][0] for i in plot_route]
    y_coords = [cities_coords[i][1] for i in plot_route]
    
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='r', alpha=0.7)
    plt.plot(x_coords[0], y_coords[0], marker='*', color='green', markersize=15, label='Start') # 标出起点
    plt.title('Best TSP Route (30 Cities)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()