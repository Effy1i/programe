import numpy as np
import random
from copy import deepcopy

# --- 1. 数据定义与初始化 (问题参数和常量定义) ---

E = ['E1', 'E2'] # 所有可能的出入口/出口选择节点列表。
MAX_FIREFIGHTERS = 15 # 潜在消防员数量上限。设定一个大于房间数的上限 (例如 10 人)，
                     # 算法将通过成本效益分析，自主决定实际出动的人数 N_F (即动态人数决策)。
F_ALL = [f'F{i}' for i in range(1, MAX_FIREFIGHTERS + 1)] # 潜在消防员的集合 (F1 到 F10)。
R = [f'R{i}' for i in range(1, 17)] # 房间集合 (R1 到 R6)。
NODES = E + R # 所有可能的节点，用于在时间矩阵中查找索引。
ROOM_COUNT = len(R) # 房间总数 (6)。
E_COUNT = len(E) # 出入口总数 (2)。

# 实例：通行时间矩阵 (分钟)
# 维度为 (len(E) + len(R)) x (len(E) + len(R)) = 8x8。
TIME_MATRIX_LIST = [
     [0.000, 0.287, 0.084, 0.180, 0.275, 0.084, 0.180, 0.275, 0.096, 0.180, 0.181, 0.324, 0.190, 0.324, 0.216, 0.359, 0.216, 0.359],  # E1
    [0.287, 0.000, 0.275, 0.179, 0.084, 0.275, 0.179, 0.084, 0.335, 0.179, 0.325, 0.180, 0.342, 0.180, 0.359, 0.215, 0.359, 0.215],  # E2
    [0.084, 0.275, 0.000, 0.096, 0.191, 0.072, 0.168, 0.263, 0.108, 0.263, 0.193, 0.335, 0.278, 0.408, 0.227, 0.371, 0.300, 0.443],  # R1
    [0.180, 0.179, 0.096, 0.000, 0.096, 0.168, 0.072, 0.167, 0.203, 0.359, 0.289, 0.287, 0.379, 0.360, 0.323, 0.323, 0.395, 0.394],  # R2
    [0.275, 0.084, 0.191, 0.096, 0.000, 0.263, 0.167, 0.072, 0.275, 0.287, 0.361, 0.215, 0.455, 0.288, 0.395, 0.251, 0.467, 0.323],  # R3
    [0.084, 0.275, 0.072, 0.168, 0.263, 0.000, 0.096, 0.192, 0.180, 0.263, 0.265, 0.407, 0.203, 0.336, 0.299, 0.443, 0.228, 0.371],  # R4
    [0.180, 0.179, 0.168, 0.072, 0.167, 0.096, 0.000, 0.096, 0.275, 0.359, 0.361, 0.359, 0.304, 0.288, 0.395, 0.395, 0.323, 0.323],  # R5
    [0.275, 0.084, 0.263, 0.167, 0.072, 0.192, 0.096, 0.000, 0.347, 0.287, 0.433, 0.287, 0.379, 0.216, 0.466, 0.323, 0.395, 0.251],  # R6
    [0.096, 0.335, 0.108, 0.203, 0.275, 0.180, 0.275, 0.347, 0.000, 0.156, 0.133, 0.276, 0.215, 0.348, 0.168, 0.311, 0.240, 0.383],  # R7
    [0.180, 0.179, 0.263, 0.359, 0.287, 0.263, 0.359, 0.287, 0.156, 0.000, 0.289, 0.287, 0.304, 0.288, 0.323, 0.323, 0.323, 0.323],  # R8
    [0.181, 0.325, 0.193, 0.289, 0.361, 0.265, 0.361, 0.433, 0.133, 0.289, 0.000, 0.145, 0.076, 0.217, 0.181, 0.325, 0.253, 0.397],  # R9
    [0.324, 0.180, 0.335, 0.287, 0.215, 0.407, 0.359, 0.287, 0.276, 0.287, 0.145, 0.000, 0.228, 0.072, 0.323, 0.179, 0.396, 0.251],  # R10
    [0.190, 0.342, 0.278, 0.379, 0.455, 0.203, 0.304, 0.379, 0.215, 0.304, 0.076, 0.228, 0.000, 0.152, 0.266, 0.417, 0.190, 0.342],  # R11
    [0.324, 0.180, 0.408, 0.360, 0.288, 0.336, 0.288, 0.216, 0.348, 0.288, 0.217, 0.072, 0.152, 0.000, 0.396, 0.252, 0.324, 0.180],  # R12
    [0.216, 0.359, 0.227, 0.323, 0.395, 0.299, 0.395, 0.466, 0.168, 0.323, 0.181, 0.323, 0.266, 0.396, 0.000, 0.144, 0.072, 0.215],  # R13
    [0.359, 0.215, 0.371, 0.323, 0.251, 0.443, 0.395, 0.323, 0.311, 0.323, 0.325, 0.179, 0.417, 0.252, 0.144, 0.000, 0.215, 0.072],  # R14
    [0.216, 0.359, 0.300, 0.395, 0.467, 0.228, 0.323, 0.395, 0.240, 0.323, 0.253, 0.396, 0.190, 0.324, 0.072, 0.215, 0.000, 0.144],  # R15
    [0.359, 0.215, 0.443, 0.394, 0.323, 0.371, 0.323, 0.251, 0.383, 0.323, 0.397, 0.251, 0.342, 0.180, 0.215, 0.072, 0.144, 0.000]   # R16
]
TIME_MATRIX = np.array(TIME_MATRIX_LIST)

# 实例：房间固有价值 V_r (房间的初始重要性，用于计算收益项)
ROOM_VALUES = {
    'R1': 100,
    'R2': 100,
    'R3': 100,
    'R4': 100,
    'R5': 100,
    'R6': 100,
    'R7': 80,
    'R8': 110,
    'R9': 140,
    'R10': 130,
    'R11': 140,
    'R12': 160,
    'R13': 120,
    'R14': 120,
    'R15': 120,
    'R16': 120
}


# 实例：房间搜查时间 S_i (搜查房间 R_i 所需的固有时间)
SEARCH_TIMES =  {
'R1': 2.300, 'R2': 2.297, 'R3': 2.295, 'R4': 2.304, 
    'R5': 2.299, 'R6': 2.297, 'R7': 1.152, 'R8': 19.539,
    'R9': 3.491, 'R10': 3.453, 'R11': 3.847, 'R12': 3.463,
    'R13': 3.450, 'R14': 3.444, 'R15': 3.457, 'R16': 3.446
}


# 固定清扫确认时间 C (完成搜查后，额外的确认时间)
CLEANUP_TIME = 10/60

# --- 目标函数参数 (影响收益 Z 的三个主要部分) ---
FIREFIGHTER_COST = 50.0  # 每名消防员出动的固有成本 C_firefighter。影响成本项。
VALUE_DECAY_RATE = 0.05  # 价值衰减系数 alpha。影响房间时效收益项。
EFFICIENCY_BONUS_BETA = 100.0 # 效率奖励系数 Beta。影响任务总效率奖励项，数值较小以保证其辅助性。

def get_time(node1, node2):
    """
    功能：根据节点名称（E1/R1等）获取通行时间 T(i, j)。
    实现：通过查找节点在 NODES 列表中的索引，从 TIME_MATRIX 中获取对应值。
    """
    i = NODES.index(node1)
    j = NODES.index(node2)
    return TIME_MATRIX[i, j]

def get_total_room_cost(room):
    """
    功能：获取房间的固定操作总时间 L_r = S_i + C (搜查时间 + 清扫确认时间)。
    """
    return SEARCH_TIMES.get(room, 0) + CLEANUP_TIME

def value_decay_function(T):
    """
    功能：计算房间价值的衰减因子 f(T)。
    公式：f(T) = max(0, 1 - alpha * T)。
    T 为房间被搜查完成的时间点 T_finish, r。因子乘 V_r 得到实际收益。
    """
    return max(0, 1 - VALUE_DECAY_RATE * T)

# --- 3. 核心函数：解码与评估 (目标函数 Z 计算) ---

def decode_and_evaluate(chromosome):
    """
    功能：将染色体解码，计算总收益 Z、实际活跃人数 N_F 和适应度。

    目标函数：Maximize Z = (房间总衰减价值) - (消防员总成本) + (效率奖励)
                Z = (Sum(V_r * f(T_finish, r))) - (N_F * C_ff) + (Beta / T_total)
    
    染色体结构: 
    [ (R_1, F_k), ...] (房间分配与路径顺序) + [F_1入口, ...] (入口选择) + [F_1出口, ...] (出口选择)
    """
    # 染色体切片：动态获取三种基因序列
    room_genes = chromosome[:ROOM_COUNT]
    entry_genes = chromosome[ROOM_COUNT : ROOM_COUNT + MAX_FIREFIGHTERS]
    exit_genes = chromosome[ROOM_COUNT + MAX_FIREFIGHTERS :] 
    
    # 1. 初始化消防员任务字典，记录每个潜在消防员的起点、终点和任务序列
    assignments = {f: {'path_seq': [], 
                       'start_node': E[entry_genes[k]], # 根据入口基因索引 (0或1) 映射到 E1/E2
                       'end_node': E[exit_genes[k]],    # 根据出口基因索引 (0或1) 映射到 E1/E2
                       'total_time': 0.0}
                   for k, f in enumerate(F_ALL)}
    
    # 根据房间基因，将房间分配到对应的消防员的任务序列中
    for room, fk in room_genes:
        assignments[fk]['path_seq'].append(room)

    total_value = 0.0          # 累计所有房间的衰减后价值 (收益项)
    active_firefighters = 0    # 实际出动的消防员人数 N_F (成本项)
    all_completion_times = []  # 用于记录所有活跃消防员的最终撤离时间 (效率项)
    
    # 2. 遍历所有潜在消防员，计算其路径、收益贡献和撤离时间
    for fk, data in assignments.items():
        path_seq = data['path_seq']
        start_node = data['start_node']
        end_node = data['end_node']
        
        # 检查消防员是否被分配了任务。若没有，则不计入活跃人数 N_F。
        if not path_seq:
            continue
            
        active_firefighters += 1 # 确认该消防员被使用
        current_time = 0.0
        current_node = start_node
        
        # 路径遍历：计算每个房间的完成时间 T_finish, r
        for room in path_seq:
            # i. 通行时间：从当前位置移动到下一个房间
            travel_time = get_time(current_node, room)
            current_time += travel_time
            
            # ii. 操作时间：房间搜查与清扫确认 L_r
            operation_time = get_total_room_cost(room)
            current_time += operation_time
            
            # iii. 价值计算：在 T_finish, r = current_time 时的衰减价值
            value = ROOM_VALUES[room]
            decay_factor = value_decay_function(current_time)
            total_value += value * decay_factor # 累加到总收益中
            
            current_node = room # 更新当前位置
        
        # iv. 撤离时间：从最后一个房间到预定出口 E 的通行时间
        travel_to_exit = get_time(current_node, end_node)
        current_time += travel_to_exit
        data['total_time'] = current_time 
        
        all_completion_times.append(current_time) # 记录该消防员的最终撤离时间 T_complete, Fk


    # 3. 计算总净收益 Z (目标函数的三部分求和)
    
    # a) 消防员总成本
    firefighter_cost = active_firefighters * FIREFIGHTER_COST
    
    # b) 效率奖励项 (Beta / T_total)
    if active_firefighters > 0:
        T_total = max(all_completion_times) # T_total 是所有活跃消防员中最晚的撤离时间
        # 设置一个防止除零或 T_total 过小的保护
        efficiency_bonus = EFFICIENCY_BONUS_BETA / (T_total + 1e-6) 
        
    else:
        # 如果没有活跃消防员，T_total 设为极高值，效率奖励为零，但净收益会被成本项主导
        T_total = 1e6 
        efficiency_bonus = 0.0

    # 最终净收益 Z = 房间总衰减价值 - 消防员总成本 + 效率奖励
    net_gain = total_value - firefighter_cost + efficiency_bonus

    fitness = net_gain # 适应度即为净收益 Z
    
    # 返回适应度、净收益、活动人数和撤离时间列表
    return fitness, net_gain, active_firefighters, all_completion_times


# --- 4. 遗传操作 (保持通用性) ---

def initialize_population(pop_size, rooms, max_f):
    """
    功能：随机初始化种群。确保房间分配、入口选择和出口选择都是随机的起始点。
    """
    population = []
    f_choices = F_ALL # 潜在消防员列表
    
    for _ in range(pop_size):
        # 房间基因：随机分配给 F1 到 Fmax_f，并随机打乱顺序
        assigned_rooms = [(r, random.choice(f_choices)) for r in rooms]
        random.shuffle(assigned_rooms)
        
        # 入口和出口基因：随机选择 E1 (0) 或 E2 (1)，数量为 MAX_FIREFIGHTERS
        entry_genes = [random.randint(0, E_COUNT - 1) for _ in range(max_f)] 
        exit_genes = [random.randint(0, E_COUNT - 1) for _ in range(max_f)] 
        
        chromosome = assigned_rooms + entry_genes + exit_genes
        population.append(chromosome)
    return population

def crossover(parent1, parent2):
    """
    功能：交叉操作。
    房间基因：使用顺序交叉 (OX Crossover) 变体，保留房间访问的相对顺序。
    入口/出口基因：使用单点交叉。
    """
    
    p1 = deepcopy(parent1)
    p2 = deepcopy(parent2)
    
    # 1. 房间基因交叉 (保证房间不重复且分配信息保留)
    room_genes_size = ROOM_COUNT
    p1_rooms = p1[:room_genes_size]
    p2_rooms = p2[:room_genes_size]
    
    # 随机选择切片位置 (P1 的切片将保留在子代中)
    start, end = sorted(random.sample(range(room_genes_size), 2))
    child_rooms = [None] * room_genes_size
    child_rooms[start:end] = p1_rooms[start:end]
    
    # 填充：按 P2 的相对顺序填充剩余部分
    p1_rooms_set = {room for room, fk in p1_rooms[start:end]}
    p2_sequence = [item for item in p2_rooms if item[0] not in p1_rooms_set]
    fill_index = end
    # 循环填充剩余的 None 位置
    for item in p2_sequence:
        if fill_index >= room_genes_size: fill_index = 0
        while child_rooms[fill_index] is not None:
            fill_index += 1
            if fill_index >= room_genes_size: fill_index = 0
        child_rooms[fill_index] = item
        fill_index += 1
    
    # 2. 入口和出口基因交叉 (单点交叉)
    entry_exit_size = MAX_FIREFIGHTERS * E_COUNT 
    p1_ends = p1[room_genes_size:]
    p2_ends = p2[room_genes_size:]
    
    # 随机选择一个交叉点，交换入口/出口基因序列
    cross_point = random.randint(1, entry_exit_size - 1) 
    child_entries_exits = p1_ends[:cross_point] + p2_ends[cross_point:]

    return child_rooms + child_entries_exits

def mutate(chromosome, mutation_rate):
    """
    功能：变异操作。引入随机扰动，探索新的解空间。
    变异类型：路径顺序、房间分配、入口和出口选择。
    """
    mutated_chromosome = deepcopy(chromosome)
    room_genes_size = ROOM_COUNT
    f_choices = F_ALL
    
    # 1. 路径顺序变异 (交换房间基因的两个位置)
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(room_genes_size), 2)
        mutated_chromosome[idx1], mutated_chromosome[idx2] = \
            mutated_chromosome[idx2], mutated_chromosome[idx1]

    # 2. 分配变异 (随机改变一个房间的分配 Fk -> F_j)
    if random.random() < mutation_rate:
        idx = random.randint(0, room_genes_size - 1)
        room, _ = mutated_chromosome[idx]
        new_fk = random.choice(f_choices) # 可以分配给任何一个潜在消防员
        mutated_chromosome[idx] = (room, new_fk)
        
    # 3. 入口和出口变异 (随机改变 E1/E2 的选择)
    start_index = room_genes_size
    total_end_genes = MAX_FIREFIGHTERS * E_COUNT
    
    for i in range(total_end_genes):
        if random.random() < mutation_rate:
            end_gene_idx = start_index + i
            # 随机在 [0, 1] 之间选择，改变入口/出口
            mutated_chromosome[end_gene_idx] = random.randint(0, E_COUNT - 1)
        
    return mutated_chromosome

# --- 5. 主遗传算法循环 ---

def genetic_algorithm_solver_max_profit(rooms, max_f, pop_size=150, generations=500, mutation_rate=0.15, elite_size=15, convergence_limit=50):
    """
    功能：遗传算法主循环，负责迭代、选择、交叉、变异和收敛检查。
    """
    population = initialize_population(pop_size, rooms, max_f)
    best_fitness = -np.inf # 目标是最大化收益，初始化最佳适应度为负无穷
    best_chromosome = None
    generations_without_improvement = 0 # 连续未改进代数计数器
    
    print(f"--- 收益最大化遗传算法 (动态人数 + 效率奖励) 参数 ---")
    print(f"  效率奖励 Beta={EFFICIENCY_BONUS_BETA}, 成本={FIREFIGHTER_COST}, 衰减率={VALUE_DECAY_RATE}")
    print(f"  种群大小={pop_size}, 最大代数={generations}, 变异率={mutation_rate}, 潜在人数={max_f}")
    
    for generation in range(generations):
        # 1. 评估适应度
        results = [(chromosome, *decode_and_evaluate(chromosome)) for chromosome in population]
        results.sort(key=lambda x: x[1], reverse=True) # 按净收益(fitness)降序排列
        
        current_best = results[0]
        
        # 2. 更新全局最佳解和收敛计数器
        if current_best[1] > best_fitness:
            best_fitness = current_best[1]
            best_chromosome = current_best[0]
            generations_without_improvement = 0 # 发现更优解，重置计数器
        else:
            generations_without_improvement += 1 # 未发现更优解，计数器增加
            
        # 3. 打印进度和收敛检查
        if generation % 50 == 0 or generation == generations - 1:
            best_net_gain = current_best[1]
            active_ff = current_best[3]
            
            # 重新解码最佳染色体，获取最晚撤离时间 T_total 用于打印
            _, _, _, completion_times = decode_and_evaluate(best_chromosome)
            T_total_best = max(completion_times) if completion_times else 0
            
            print(f"Gen {generation:03d}: Net Gain={best_net_gain:.2f} (Active FF:{active_ff}, T_total:{T_total_best:.1f})")

        # 提前终止检查：如果连续 N 代净收益没有改善，则停止迭代
        if generations_without_improvement >= convergence_limit:
            print(f"\n📢 算法在第 {generation} 代收敛。连续 {convergence_limit} 代未发现改进。提前终止。")
            break
            
        # 4. 选择、交叉与变异 (生成下一代种群)
        new_population = [r[0] for r in results[:elite_size]] # 精英保留策略
        
        # 计算轮盘赌权重：适应度为负数（亏损）的个体权重设为 0
        total_fitness = sum(max(0, r[1]) for r in results) 
        
        if total_fitness < 1e-6:
             # 如果所有收益都接近负数，则平均选择，继续探索
             selection_probabilities = [1/pop_size] * pop_size
        else:
            # 标准轮盘赌选择，基于非负的适应度分配概率
            selection_probabilities = [max(0, r[1]) / total_fitness for r in results]
        
        while len(new_population) < pop_size:
            # 根据权重概率选择两个父代进行交叉
            parents = random.choices(results, weights=selection_probabilities, k=2)
            parent1 = parents[0][0]
            parent2 = parents[1][0]
            
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
            
        population = new_population # 更新种群

    # 最终评估最佳染色体
    final_fitness, final_net_gain, final_active_ff, final_times = decode_and_evaluate(best_chromosome)
    
    return final_net_gain, best_chromosome, final_active_ff, final_times


# --- 6. 运行主程序 ---

if __name__ == '__main__':
    random.seed(42) # 设置随机种子以保证实验结果可复现

    final_net_gain, final_chromosome, active_ff, final_times = genetic_algorithm_solver_max_profit(
        R, MAX_FIREFIGHTERS, convergence_limit=50
    )

    # --- 最终结果展示与解码 ---
    room_genes = final_chromosome[:ROOM_COUNT]
    entry_genes = final_chromosome[ROOM_COUNT : ROOM_COUNT + MAX_FIREFIGHTERS]
    exit_genes = final_chromosome[ROOM_COUNT + MAX_FIREFIGHTERS :]

    # 计算最终的最晚撤离时间 (T_total) 和效率奖励值
    T_total_max = max(final_times) if final_times else 0
    efficiency_bonus_value = EFFICIENCY_BONUS_BETA / T_total_max if T_total_max > 0 else 0
    
    print("\n### 遗传算法最终结果 (收益最大化 - 纳入 T_total 效率奖励) ###")
    print(f"**🔥 最终总净收益 (Z): {final_net_gain:.2f}**")
    print(f"出动消防员数量: {active_ff}")
    print(f"最晚撤离时间 (T_total): {T_total_max:.1f} 分钟")
    print(f"效率奖励项 (Beta/T_total): +{efficiency_bonus_value:.2f}")
    print(f"总消防员固有成本: -{active_ff * FIREFIGHTER_COST:.2f}")
    print("-" * 50)
    
    # 详细打印每个出动消防员的路径信息、耗时和价值贡献
    for k in range(MAX_FIREFIGHTERS):
        fk = F_ALL[k]
        start_node = E[entry_genes[k]]
        end_node = E[exit_genes[k]]
        
        path_rooms = [r for r, f_assign in room_genes if f_assign == fk]
        
        if path_rooms:
            # 重新计算一次，获取该消防员的精确贡献和耗时
            current_time = 0.0
            current_node = start_node
            total_value_fk = 0
            
            for room in path_rooms:
                 travel_time = get_time(current_node, room)
                 current_time += travel_time
                 operation_time = get_total_room_cost(room)
                 current_time += operation_time
                 
                 decay_factor = value_decay_function(current_time)
                 value = ROOM_VALUES[room] * decay_factor
                 total_value_fk += value # 累加该消防员在房间价值中的贡献

                 current_node = room
            
            travel_to_exit = get_time(current_node, end_node)
            current_time += travel_to_exit # 完整路径时间 = 任务时间 + 撤离时间
                 
            path_str = f"{start_node} -> {' -> '.join(path_rooms)} -> {end_node}"
            
            print(f"[{fk} - 活跃] 入口:{start_node}, 出口:{end_node}")
            print(f"  完整路径耗时: {current_time:.1f} 分钟 (包括撤离)")
            print(f"  路径: {path_str}")
            print(f"  贡献衰减总价值: {total_value_fk:.2f}")
        # 否则，该消防员未出动