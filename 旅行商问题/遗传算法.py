import numpy as np
import matplotlib.pyplot as plt
import random

# ================= 1. 基础工具函数 =================

def calculate_total_distance(route, dist_matrix):
    """计算一条完整路径的总距离"""
    dist = 0
    n = len(route)
    for i in range(n):
        dist += dist_matrix[route[i]][route[(i + 1) % n]]
    return dist

# ================= 2. 遗传算法核心算子 (繁衍与变异) =================

def order_crossover(parent1, parent2):
    """
    【核心魔法】顺序交叉法 (Order Crossover, OX)
    专门用于处理“全排列”问题的交叉，保证城市不重复、不遗漏。
    """
    n = len(parent1)
    # 1. 随机选定要从爸爸那里继承的基因片段起止点
    start, end = sorted(random.sample(range(n), 2))
    
    child = [-1] * n
    
    # 2. 原封不动地把爸爸的这段基因复制给孩子
    child[start:end+1] = parent1[start:end+1]
    
    # 3. 把妈妈的基因按顺序填入孩子剩下的空位中 (跳过已经有的城市)
    p2_idx = 0
    for i in range(n):
        if child[i] == -1: # 找到一个空位
            # 在妈妈的基因里找一个孩子还没去过的城市
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1
            
    return child

def mutate(route, mutation_rate=0.1):
    """
    基因突变：以极小的概率，随机交换路线中的两个城市 (Swap)
    """
    new_route = route.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

'''
def mutate(route, dist_matrix, mutation_rate=0.1):
    """
    【模因算法核心】结合模拟退火的变异环节 (Memetic Mutation)
    触发变异时，不再只做一次盲目的交换，而是对当前路线进行一次微型模拟退火(Micro-SA)搜索。
    """
    new_route = route.copy()
    
    # 以一定概率触发变异（局部搜索）
    if random.random() < mutation_rate:
        current_cost = calculate_total_distance(new_route, dist_matrix)
        
        # 设定微型模拟退火的参数 (快进快出)
        T = 100.0       # 初始温度
        min_T = 1.0     # 终止温度
        alpha = 0.85    # 极快的降温速率，避免在此处消耗过多算力
        
        while T > min_T:
            neighbor_route = new_route.copy()
            
            # 随机决定使用 Swap 还是 2-Opt 进行扰动
            if random.random() < 0.5:
                i, j = random.sample(range(len(neighbor_route)), 2)
                neighbor_route[i], neighbor_route[j] = neighbor_route[j], neighbor_route[i]
            else:
                # 加入 2-Opt 翻转，极其克制路线交叉
                i, j = sorted(random.sample(range(len(neighbor_route)), 2))
                neighbor_route[i:j+1] = reversed(neighbor_route[i:j+1])
                
            new_cost = calculate_total_distance(neighbor_route, dist_matrix)
            delta = new_cost - current_cost
            
            # Metropolis 准则
            if delta < 0 or random.random() < math.exp(-delta / T):
                new_route = neighbor_route
                current_cost = new_cost
                
            T *= alpha # 快速降温   
    return new_route
'''

# ================= 3. 遗传算法主引擎 =================

def ga_tsp(dist_matrix, pop_size=100, generations=300, mutation_rate=0.1):
    n = len(dist_matrix)
    
    # 1. 初始化种群 (生成 100 只随机的猴子/路线)
    population = []
    for _ in range(pop_size):
        route = list(range(n))
        random.shuffle(route) # 随机打乱
        dist = calculate_total_distance(route, dist_matrix)
        population.append({'route': route, 'distance': dist})
        
    # 记录历史最优
    best_ind = min(population, key=lambda x: x['distance'])
    history_best_dist = [best_ind['distance']]
    
    print(f"🌍 创世纪！初代最强路线距离: {best_ind['distance']:.2f}")
    
    # 2. 开始时代更迭 (进化迭代)
    for gen in range(generations):
        new_population = []
        
        # 【精英保留策略 (Elitism)】
        # 把这一代最优秀的 2 个个体直接保送到下一代，防止好基因在交叉变异中意外丢失
        elites = sorted(population, key=lambda x: x['distance'])[:2]
        new_population.extend(elites)
        
        # 繁衍出剩下的 98 个孩子
        while len(new_population) < pop_size:
            # 锦标赛选择 (Tournament Selection)：随机抓 3 只猴子，选出最强的当父母
            parent1 = min(random.sample(population, 3), key=lambda x: x['distance'])['route']
            parent2 = min(random.sample(population, 3), key=lambda x: x['distance'])['route']
            
            # 父母交叉生下孩子
            child_route = order_crossover(parent1, parent2)
            
            # 孩子发生极小概率的基因突变
            child_route = mutate(child_route, mutation_rate)
            
            # 计算孩子的能力 (距离)
            child_dist = calculate_total_distance(child_route, dist_matrix)
            new_population.append({'route': child_route, 'distance': child_dist})
            
        population = new_population
        
        # 记录本代最强
        current_best = min(population, key=lambda x: x['distance'])
        if current_best['distance'] < best_ind['distance']:
            best_ind = current_best
            
        history_best_dist.append(best_ind['distance'])
        print(f"  ⚡ 第 {gen+1} 代: 当前探明最短距离 = {best_ind['distance']:.2f}")
            
    return best_ind['route'], best_ind['distance'], history_best_dist

# ================= 4. 测试与可视化 =================
if __name__ == "__main__":
    # 使用与模拟退火、蚁群算法完全相同的 30 城市数据集，保证公平对比
    np.random.seed(42)
    num_cities = 30
    cities_coords = np.random.rand(num_cities, 2) * 100
    
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(cities_coords[i] - cities_coords[j])
            
    print("开始运行遗传算法 (GA) 求解 TSP...")
    
    best_route, best_cost, history = ga_tsp(
        dist_matrix, 
        pop_size=100,       # 种群数量 (猴子总数)
        generations=300,    # 繁衍代数
        mutation_rate=0.15  # 突变率 (稍微调高点有助于跳出局部最优)
    )
    
    print(f"\n🏆 GA 进化结束！最终最短距离: {best_cost:.2f}")
    
    # ======= 画图 =======
    plt.figure(figsize=(12, 5))
    
    # 图 1：收敛曲线
    plt.subplot(1, 2, 1)
    plt.plot(history, color='green', linewidth=2)
    plt.title('Genetic Algorithm Convergence Curve')
    plt.xlabel('Generations')
    plt.ylabel('Best Distance')
    plt.grid(True)
    
    # 图 2：最终路径轨迹
    plt.subplot(1, 2, 2)
    plot_route = best_route + [best_route[0]]
    x_coords = [cities_coords[i][0] for i in plot_route]
    y_coords = [cities_coords[i][1] for i in plot_route]
    
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='red', alpha=0.7)
    plt.plot(x_coords[0], y_coords[0], marker='*', color='gold', markersize=15, label='Start') # 起点
    plt.title(f'GA Best TSP Route (Cost: {best_cost:.2f})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()