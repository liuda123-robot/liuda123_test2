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

# ================= 2. 随机邻域动作 (只生成1个新解) =================

def get_random_swap(route):
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def get_random_insert(route):
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    city = new_route.pop(i)
    new_route.insert(j, city)
    return new_route

def get_random_2opt(route):
    new_route = route.copy()
    i, j = sorted(random.sample(range(len(route)), 2))
    new_route[i:j+1] = reversed(new_route[i:j+1])
    return new_route

# ================= 3. 你构思的 SA + VND 完美融合引擎 =================

def sa_vnd_structured(dist_matrix, initial_temp=500, cooling_rate=0.98, min_temp=0.1, iters_per_neighborhood=100):
    n = len(dist_matrix)
    current_route = list(range(n))
    current_cost = calculate_total_distance(current_route, dist_matrix)
    
    best_route = current_route.copy()
    best_cost = current_cost
    history_cost = []
    
    actions = [get_random_swap, get_random_insert, get_random_2opt]
    
    T = initial_temp
    print(f"初始路线代价: {current_cost:.2f}")
    
    while T > min_temp:
        k = 0 
        
        while k < len(actions):
            action = actions[k]
            improved = False 
            
            for _ in range(iters_per_neighborhood):
                new_route = action(current_route)
                new_cost = calculate_total_distance(new_route, dist_matrix)
                
                delta_e = new_cost - current_cost
                
                if delta_e < 0:
                    current_route = new_route
                    current_cost = new_cost
                    
                    # 【核心修复点】：
                    # 只有打破了全局最优记录，才算是真正的“实质性突破”，才允许重置 k！
                    # 如果只是退火接受劣解后的小幅回升，不作为重置 k 的理由。
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_route = current_route.copy()
                        improved = True  # 门槛提高，杜绝无限死循环
                        
                elif random.random() < math.exp(-delta_e / T):
                    # 接受劣解，继续探索，但不算作 improved
                    current_route = new_route
                    current_cost = new_cost
            
            # --- VND 核心切换逻辑 ---
            if improved:
                k = 0 
            else:
                k += 1 
                
        history_cost.append(best_cost)
        T *= cooling_rate 
        
    return best_route, best_cost, history_cost

# ================= 4. 测试与可视化 =================
if __name__ == "__main__":
    np.random.seed(42)
    num_cities = 200 
    cities_coords = np.random.rand(num_cities, 2) * 100
    
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(cities_coords[i] - cities_coords[j])
            
    print("开始运行 结构化 SA-VND 混合算法...")
    best_route, best_cost, history = sa_vnd_structured(
        dist_matrix, 
        initial_temp=500, 
        cooling_rate=0.98, # 稍微加快一点降温，因为内层探索很充分
        min_temp=0.1, 
        iters_per_neighborhood=300 # 每种招式在当前温度下探索 100 次
    )
    
    print(f"\n🏆 SA-VND 最终最短距离: {best_cost:.2f}")
    
    # ======= 画图 =======
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history, color='orange', linewidth=2)
    plt.title('Structured SA-VND Convergence')
    plt.xlabel('Temperature Steps')
    plt.ylabel('Best Distance')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plot_route = best_route + [best_route[0]]
    x_coords = [cities_coords[i][0] for i in plot_route]
    y_coords = [cities_coords[i][1] for i in plot_route]
    
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='red', alpha=0.7)
    plt.plot(x_coords[0], y_coords[0], marker='*', color='green', markersize=15, label='Start')
    plt.title(f'SA-VND Best Route - Cost: {best_cost:.2f}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()