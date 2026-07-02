import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ================= 1. 基础工具函数 =================

def calculate_total_distance(route, dist_matrix):
    dist = 0
    n = len(route)
    for i in range(n):
        dist += dist_matrix[route[i]][route[(i + 1) % n]]
    return dist

# ================= 2. 随机邻域动作 (不再穷举，每次只随机抽 1 个！) =================

def get_random_swap(route):
    """随机交换两点"""
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def get_random_insert(route):
    """随机单点插入"""
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    city = new_route.pop(i)
    new_route.insert(j, city)
    return new_route

def get_random_2opt(route):
    """随机两点翻转 (极度克制交叉线)"""
    new_route = route.copy()
    # 保证 i 在 j 前面
    i, j = sorted(random.sample(range(len(route)), 2))
    new_route[i:j+1] = reversed(new_route[i:j+1])
    return new_route

# ================= 3. SA-VNS 混合算法引擎 =================

def sa_vns_hybrid(dist_matrix, initial_temp=1000, cooling_rate=0.99, min_temp=0.1, iters_per_temp=100):
    n = len(dist_matrix)
    current_route = list(range(n))
    current_cost = calculate_total_distance(current_route, dist_matrix)
    
    best_route = current_route.copy()
    best_cost = current_cost
    history_cost = []
    
    # 将三大招式放入武器库
    actions = [get_random_swap, get_random_insert, get_random_2opt]
    
    T = initial_temp
    
    print(f"初始路线代价: {current_cost:.2f}")
    
    while T > min_temp:
        for _ in range(iters_per_temp):
            # 【核心魔法】：每次随机抽取一种招式来生成新路线！
            # 这种做法极大地丰富了搜索空间，并且完美避开了穷举的算力消耗
            action = random.choice(actions)
            new_route = action(current_route)
            new_cost = calculate_total_distance(new_route, dist_matrix)
            
            delta_e = new_cost - current_cost
            
            # Metropolis 准则 (模拟退火的灵魂)
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_route = new_route
                current_cost = new_cost
                
                # 记录全局最优
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_route = current_route.copy()
                    
        history_cost.append(best_cost)
        T *= cooling_rate # 降温
        
    return best_route, best_cost, history_cost

# ================= 4. 测试与可视化 =================
if __name__ == "__main__":
    np.random.seed(42)
    num_cities = 200 # 你可以尝试把它改成 100 甚至 200，纯 VND 会卡死，但这套代码依然秒出结果！
    cities_coords = np.random.rand(num_cities, 2) * 100
    
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(cities_coords[i] - cities_coords[j])
            
    print("开始运行 SA-VNS 混合算法...")
    # 注意：因为没有了极其耗时的穷举，我们可以放心地把迭代次数拉高
    best_route, best_cost, history = sa_vns_hybrid(
        dist_matrix, 
        initial_temp=500, 
        cooling_rate=0.98, 
        min_temp=0.1, 
        iters_per_temp=300 # 增加内循环次数
    )
    
    print(f"\n🏆 SA-VNS 最终最短距离: {best_cost:.2f}")
    
    # ======= 画图 =======
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history, color='purple', linewidth=2)
    plt.title('SA-VNS Convergence Curve')
    plt.xlabel('Temperature Steps')
    plt.ylabel('Best Distance')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plot_route = best_route + [best_route[0]]
    x_coords = [cities_coords[i][0] for i in plot_route]
    y_coords = [cities_coords[i][1] for i in plot_route]
    
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='red', alpha=0.7)
    plt.plot(x_coords[0], y_coords[0], marker='*', color='green', markersize=15, label='Start')
    plt.title(f'SA-VNS Best Route - Cost: {best_cost:.2f}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()