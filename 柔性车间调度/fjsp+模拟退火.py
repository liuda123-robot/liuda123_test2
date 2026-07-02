import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math

# ================= 1. 定义加工环境 (FJSP 数据字典) =================
PT = {
    (0, 0): {0: 3, 1: 4},       
    (0, 1): {0: 5, 1: 2},       
    (1, 0): {0: 2, 2: 3},       
    (1, 1): {0: 5, 1: 2},       
    (1, 2): {1: 1, 2: 4},       
    (2, 0): {0: 1, 1: 3},       
    (2, 1): {1: 4, 2: 2}        
}

op_to_global_idx = {
    (0, 0): 0, (0, 1): 1, 
    (1, 0): 2, (1, 1): 3, (1, 2): 4, 
    (2, 0): 5, (2, 1): 6
}
# 反向映射表：为了算法改 MS 码时，知道改的是哪个工序，从而去查它能用哪些机器
global_to_op = {v: k for k, v in op_to_global_idx.items()}

# ================= 2. 解码器引擎 (无需任何改动，直接复用) =================
def decode_fjsp(OS, MS, pt_dict, op_map):
    job_op_counts = {0: 0, 1: 0, 2: 0}
    machine_free_time = {0: 0, 1: 0, 2: 0}
    job_free_time = {0: 0, 1: 0, 2: 0}
    schedule = []
    
    for job_id in OS:
        op_id = job_op_counts[job_id]
        global_idx = op_map[(job_id, op_id)]
        machine_id = MS[global_idx]
        
        duration = pt_dict[(job_id, op_id)][machine_id]
        start_time = max(machine_free_time[machine_id], job_free_time[job_id])
        end_time = start_time + duration
        
        schedule.append({
            'Job': job_id, 'Op': op_id, 'Machine': machine_id,
            'Start': start_time, 'End': end_time
        })
        
        machine_free_time[machine_id] = end_time
        job_free_time[job_id] = end_time
        job_op_counts[job_id] += 1
        
    return schedule, max(machine_free_time.values())

# ================= 3. 算法核心：邻域动作 (如何改变数组) =================
def generate_neighbor(OS, MS):
    """邻域动作：随机扰动 OS 数组 或 MS 数组来生成新解"""
    new_OS = OS.copy()
    new_MS = MS.copy()
    
    # 50% 概率改变加工顺序 (动 OS)
    if random.random() < 0.5:
        i, j = random.sample(range(len(new_OS)), 2)
        new_OS[i], new_OS[j] = new_OS[j], new_OS[i]
        
    # 50% 概率改变机器选择 (动 MS)
    else:
        idx = random.randint(0, len(new_MS) - 1)
        job, op = global_to_op[idx]
        # 拿到这道工序可以用的所有机器
        available_machines = list(PT[(job, op)].keys())
        
        if len(available_machines) > 1:
            # 把它现在的机器踢出候选名单，逼迫它换一台新机器
            current_machine = new_MS[idx]
            available_machines.remove(current_machine)
            new_MS[idx] = random.choice(available_machines)
            
    return new_OS, new_MS

# ================= 4. 模拟退火优化引擎 (SA) =================
def fjsp_simulated_annealing(initial_temp=100, min_temp=0.1, cooling_rate=0.95, iters_per_temp=50):
    # 1. 瞎编一个初始合法解
    # OS 初始: 把各个工件的编号按次数平铺 [0,0, 1,1,1, 2,2]
    current_OS = [0, 0, 1, 1, 1, 2, 2]
    random.shuffle(current_OS) # 随便打乱一下作为开局
    
    # MS 初始: 给每道工序随机选一台它能用的机器
    current_MS = []
    for i in range(7):
        job, op = global_to_op[i]
        current_MS.append(random.choice(list(PT[(job, op)].keys())))
        
    # 算一下开局的耗时
    _, current_makespan = decode_fjsp(current_OS, current_MS, PT, op_to_global_idx)
    
    best_OS, best_MS = current_OS.copy(), current_MS.copy()
    best_makespan = current_makespan
    history = []
    
    print(f"🔥 退火开始！初始瞎排的总耗时: {current_makespan} 分钟")
    
    T = initial_temp
    while T > min_temp:
        for _ in range(iters_per_temp):
            # 获取新解 (换机器 或 换顺序)
            new_OS, new_MS = generate_neighbor(current_OS, current_MS)
            _, new_makespan = decode_fjsp(new_OS, new_MS, PT, op_to_global_idx)
            
            delta_e = new_makespan - current_makespan
            
            # Metropolis 准则
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_OS, current_MS = new_OS, new_MS
                current_makespan = new_makespan
                
            # 记录全局最优解
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_OS, best_MS = current_OS.copy(), current_MS.copy()
                print(f"当前最优解：{best_makespan}")
                
        history.append(best_makespan)
        T *= cooling_rate
        
    return best_OS, best_MS, best_makespan, history

# ================= 5. 甘特图绘制 (复用) =================
def draw_gantt(schedule, makespan):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {0: '#ff9999', 1: '#66b3ff', 2: '#99ff99'}
    
    for task in schedule:
        job, op, m_id, start, duration = task['Job'], task['Op'], task['Machine'], task['Start'], task['End'] - task['Start']
        ax.broken_barh([(start, duration)], (m_id - 0.4, 0.8), facecolors=colors[job], edgecolor='black', linewidth=1)
        ax.text(start + duration/2, m_id, f'J{job}-{op}', ha='center', va='center', color='black', fontweight='bold')

    ax.set_ylim(-1, 3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Machine 0', 'Machine 1', 'Machine 2'])
    ax.set_xlabel('Time (Minutes)')
    ax.set_title(f'FJSP Optimized Gantt Chart (Makespan = {makespan} mins)')
    ax.legend(handles=[mpatches.Patch(color=colors[i], label=f'Job {i}') for i in range(3)], loc='upper right')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ================= 6. 运行主程序 =================
if __name__ == "__main__":
    best_OS, best_MS, best_makespan, history = fjsp_simulated_annealing()
    
    print(f"\n🏆 退火结束！算法找到的最优解耗时: {best_makespan} 分钟")
    print(f"最牛的 OS 码 (工序排序): {best_OS}")
    print(f"最牛的 MS 码 (机器选择): {best_MS}")
    
    # 用最牛的码去生成排程表
    final_schedule, _ = decode_fjsp(best_OS, best_MS, PT, op_to_global_idx)
    draw_gantt(final_schedule, best_makespan)