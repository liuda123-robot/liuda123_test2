import numpy as np
import matplotlib.pyplot as plt
import random

# ================= 1. 蚁群算法核心引擎 =================

class AntColonyTSP:
    def __init__(self, dist_matrix, num_ants, num_iterations, alpha=1.0, beta=2.0, rho=0.1, Q=100):
        """
        :param dist_matrix: 距离矩阵
        :param num_ants: 蚂蚁数量 (通常与城市数量相当)
        :param num_iterations: 迭代代数
        :param alpha: 信息素重要程度 (跟风倾向)
        :param beta: 启发函数重要程度 (贪心倾向，通常设大一点，优先看距离)
        :param rho: 信息素挥发系数 (0~1)
        :param Q: 信息素常数 (决定释放量的大小)
        """
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # 启发信息矩阵 (Eta): 距离的倒数，距离越近，启发值越大
        # 为了避免除以 0，对角线（自己到自己）设为无穷大或 0
        self.eta = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.eta[i][j] = 1.0 / self.dist_matrix[i][j]
                    
        # 信息素矩阵 (Pheromone/Tau): 初始化为极小的常数
        self.tau = np.ones((self.num_cities, self.num_cities)) * 0.1
        
    def solve(self):
        best_route = None
        best_cost = float('inf')
        history_best_cost = []
        
        print(f"开始蚁群出征！共 {self.num_iterations} 代，每代 {self.num_ants} 只蚂蚁...")
        
        for iteration in range(self.num_iterations):
            all_routes = []
            all_costs = []
            
            # 1. 每一只蚂蚁开始构建自己的路径
            for ant in range(self.num_ants):
                route = self.construct_path()
                cost = self.calculate_cost(route)
                all_routes.append(route)
                all_costs.append(cost)
                
                # 记录全局最优
                if cost < best_cost:
                    best_cost = cost
                    best_route = route.copy()
            
            # 2. 全局信息素更新 (所有蚂蚁走完后结算)
            self.update_pheromones(all_routes, all_costs)
            
            history_best_cost.append(best_cost)
            
            if (iteration + 1) % 10 == 0:
                print(f"  第 {iteration + 1} 代: 当前探明最短距离 = {best_cost:.2f}")
                
        return best_route, best_cost, history_best_cost
        
    def construct_path(self):
        """单只蚂蚁构建路径的过程"""
        # 随机选择一个出生城市
        start_city = random.randint(0, self.num_cities - 1)
        route = [start_city]
        visited = set([start_city])
        
        current_city = start_city
        
        # 走到所有城市都被访问过
        while len(visited) < self.num_cities:
            # 计算去各个未访问城市的概率
            probabilities = []
            unvisited_cities = []
            
            for next_city in range(self.num_cities):
                if next_city not in visited:
                    # 核心公式：概率因子 = (信息素^alpha) * (能见度^beta)
                    tau_val = self.tau[current_city][next_city] ** self.alpha
                    eta_val = self.eta[current_city][next_city] ** self.beta
                    probabilities.append(tau_val * eta_val)
                    unvisited_cities.append(next_city)
            
            # 轮盘赌选择 (Roulette Wheel Selection)
            # 概率归一化
            prob_sum = sum(probabilities)
            normalized_probs = [p / prob_sum for p in probabilities]
            
            # 根据概率随机挑选下一个城市 (信息素浓、距离近的概率极大)
            chosen_city = np.random.choice(unvisited_cities, p=normalized_probs)
            
            route.append(chosen_city)
            visited.add(chosen_city)
            current_city = chosen_city
            
        return route
        
    def calculate_cost(self, route):
        dist = 0
        for i in range(self.num_cities):
            dist += self.dist_matrix[route[i]][route[(i + 1) % self.num_cities]]
        return dist
        
    def update_pheromones(self, all_routes, all_costs):
        """信息素挥发与沉积"""
        # 1. 挥发: 所有路上的气味先散去一部分
        self.tau = (1 - self.rho) * self.tau
        
        # 2. 沉积: 让这一代的蚂蚁撒下新的气味
        for i in range(self.num_ants):
            route = all_routes[i]
            cost = all_costs[i]
            
            # 撒气味的规则：这只蚂蚁走的总路程越短，它在路上留下的气味就越浓！(Q/L)
            pheromone_to_add = self.Q / cost 
            
            for j in range(self.num_cities):
                city_a = route[j]
                city_b = route[(j + 1) % self.num_cities]
                # 无向图，双向更新
                self.tau[city_a][city_b] += pheromone_to_add
                self.tau[city_b][city_a] += pheromone_to_add

# ================= 2. 测试与可视化 =================
if __name__ == "__main__":
    np.random.seed(42)
    num_cities = 30 
    cities_coords = np.random.rand(num_cities, 2) * 100
    
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(cities_coords[i] - cities_coords[j])
            
    # 初始化蚁群算法
    # 调参技巧：beta 设为 2~5 能让蚂蚁更倾向于走近路，收敛极快；rho=0.1 代表每次挥发 10%
    aco = AntColonyTSP(dist_matrix, num_ants=30, num_iterations=100, alpha=1.0, beta=3.0, rho=0.1)
    
    best_route, best_cost, history = aco.solve()
    
    print(f"\n🏆 蚁群算法 最终最短距离: {best_cost:.2f}")
    
    # ======= 画图 =======
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history, color='teal', linewidth=2)
    plt.title('ACO Convergence Curve')
    plt.xlabel('Generations')
    plt.ylabel('Best Distance')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plot_route = best_route + [best_route[0]]
    x_coords = [cities_coords[i][0] for i in plot_route]
    y_coords = [cities_coords[i][1] for i in plot_route]
    
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='red', alpha=0.7)
    plt.plot(x_coords[0], y_coords[0], marker='*', color='green', markersize=15, label='Start')
    plt.title(f'ACO Best Route - Cost: {best_cost:.2f}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()