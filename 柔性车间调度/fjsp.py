import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================= 1. 定义工厂的加工环境 (FJSP数据) =================
# PT (Processing Time): 记录每道工序在不同机器上的加工时间
# 格式: (工件ID, 工序序号): {可用机器ID: 耗时, 可用机器ID: 耗时}
PT = {
    (0, 0): {0: 3, 1: 4},       # 工件0-工序0: 在M0需3分钟，在M1需4分钟
    (0, 1): {0: 5, 1: 2},       # 工件0-工序1: 在M0需5分钟，在M1需2分钟
    (1, 0): {0: 2, 2: 3},       # 工件1-工序0: 在M0需2分钟，在M2需3分钟
    (1, 1): {0: 5, 1: 2},       # 工件1-工序1: 在M0需5分钟，在M1需2分钟
    (1, 2): {1: 1, 2: 4},       # 工件1-工序2: 在M1需1分钟，在M2需4分钟
    (2, 0): {0: 1, 1: 3},       # 工件2-工序0: 在M0需1分钟，在M1需3分钟
    (2, 1): {1: 4, 2: 2}        # 工件2-工序1: 在M1需4分钟，在M2需2分钟
}

# 建立全局工序索引的映射，为了方便在 MS 数组中查找机器
# 按照 (0,0), (0,1), (1,0), (1,1)... 的平铺顺序
op_to_global_idx = {
    (0, 0): 0, (0, 1): 1, 
    (1, 0): 2, (1, 1): 3, (1, 2): 4, 
    (2, 0): 5, (2, 1): 6
}

# ================= 2. 算法生成的双层染色体 (待解码) =================
# OS码 (Operation Sequence): [1, 0, 1, 2, 0, 1, 2]
# 包含两个 0，三个 1，两个 2，代表它们的工序数
OS_array = [1, 0, 1, 2, 0, 1, 2]


# MS码 (Machine Selection): 代表 7 道全局工序分别选哪台机器
# (直接存放真实机器的ID，方便理解)
MS_array = [0, 1, 2, 1, 2, 0, 2] 

# ================= 3. 核心解码引擎 =================
def decode_fjsp(OS, MS, pt_dict, op_map):
    """将 OS 和 MS 数组解码成具体的排产时刻表"""
    
    # 记录每个工件目前进行到第几道工序了
    job_op_counts = {0: 0, 1: 0, 2: 0}
    
    # 记录【每一台机器】什么时候空闲
    machine_free_time = {0: 0, 1: 0, 2: 0}
    
    # 记录【每一个工件】什么时候空闲 (绝不能同时在两台机器上加工)
    job_free_time = {0: 0, 1: 0, 2: 0}
    
    # 存放最终的排程结果
    # 格式: [{'Job': 0, 'Op': 0, 'Machine': 0, 'Start': 0, 'End': 3}, ...]
    schedule = []
    
    # 【最核心魔法：从左到右扫描 OS 数组】
    for job_id in OS:
        # 1. 确定当前是该工件的第几道工序
        op_id = job_op_counts[job_id]
        
        # 2. 查表获取全局索引，并去 MS 数组中看它被分配到了哪台机器
        global_idx = op_map[(job_id, op_id)]
        machine_id = MS[global_idx]
        
        # 3. 查加工时间表，获取耗时
        duration = pt_dict[(job_id, op_id)][machine_id]
        
        # 4. 计算开工时间 (核心逻辑：必须等机器空闲，且工件也空闲，才能开工)
        start_time = max(machine_free_time[machine_id], job_free_time[job_id])
        end_time = start_time + duration
        
        # 5. 记录到排程表中
        schedule.append({
            'Job': job_id,
            'Op': op_id,
            'Machine': machine_id,
            'Start': start_time,
            'End': end_time
        })
        
        # 6. 更新状态，准备迎接下一个任务
        machine_free_time[machine_id] = end_time
        job_free_time[job_id] = end_time
        job_op_counts[job_id] += 1  # 工序计数器往前走一步
        
    return schedule, max(machine_free_time.values()) # 返回详细时刻表和总完工时间 (Makespan)

# ================= 4. 甘特图可视化 =================
def draw_gantt(schedule, makespan):
    """使用 matplotlib 绘制精美的甘特图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 为不同的工件分配不同的颜色
    colors = {0: '#ff9999', 1: '#66b3ff', 2: '#99ff99'}
    
    # 解析排程结果并画长方形 (Bar)
    for task in schedule:
        job = task['Job']
        op = task['Op']
        m_id = task['Machine']
        start = task['Start']
        duration = task['End'] - task['Start']
        
        # 画矩形块 (x起始, 持续时间), (y起始, 宽度)
        # y 轴对应机器 0, 1, 2
        ax.broken_barh([(start, duration)], (m_id - 0.4, 0.8), 
                       facecolors=colors[job], edgecolor='black', linewidth=1)
        
        # 在色块中间写上文字 (J0-0 代表工件0工序0)
        ax.text(start + duration/2, m_id, f'J{job}-{op}', 
                ha='center', va='center', color='black', fontweight='bold')

    # 设置图表格式
    ax.set_ylim(-1, 3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Machine 0', 'Machine 1', 'Machine 2'])
    ax.set_xlabel('Time (Minutes)')
    ax.set_title(f'FJSP Gantt Chart (Makespan = {makespan} mins)')
    
    # 添加图例
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Job {i}') for i in range(3)]
    ax.legend(handles=legend_patches, loc='upper right')
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ================= 5. 运行主程序 =================
if __name__ == "__main__":
    print("正在解码 OS 和 MS 数组...")
    final_schedule, total_makespan = decode_fjsp(OS_array, MS_array, PT, op_to_global_idx)
    
    print("\n--- 最终排程结果 ---")
    for task in final_schedule:
        print(f"工件 {task['Job']} 的工序 {task['Op']} -> 机器 {task['Machine']} | "
              f"时间段: {task['Start']} 至 {task['End']}")
              
    print(f"\n总完工时间 (Makespan): {total_makespan} 分钟")
    
    print("\n正在生成甘特图...")
    draw_gantt(final_schedule, total_makespan)