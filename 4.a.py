import random
import math
import numpy as np
from collections import defaultdict, namedtuple
import time
import heapq
from concurrent.futures import ProcessPoolExecutor
import os

# 定义环境节点类型
NODE_TYPE = namedtuple('NodeType', ['name', 'type', 'is_entry', 'is_exit'])
# 节点：6个房间 (R1-R6), 3个过道 (C1-C3), 2个入口/出口 (E1, E2)
NODES = {
'R1': NODE_TYPE('R1', 'Room', False, False),
'R2': NODE_TYPE('R2', 'Room', False, False),
'R3': NODE_TYPE('R3', 'Room', False, False),
'R4': NODE_TYPE('R4', 'Room', False, False),
'R5': NODE_TYPE('R5', 'Room', False, False),
'R6': NODE_TYPE('R6', 'Room', False, False),
'C1': NODE_TYPE('C1', 'Corridor', False, False),
'C2': NODE_TYPE('C2', 'Corridor', False, False),
'C3': NODE_TYPE('C3', 'Corridor', False, False),
'E1': NODE_TYPE('E1', 'Entry', True, True),
'E2': NODE_TYPE('E2', 'Entry', True, True),
}
# 拓扑或物理距离 D(i, j) - 使用邻接表表示，距离越小，移动越快
# 假设距离单位为米
DISTANCES = {
'R1': {'C1': 3},
'R2': {'C2': 3},
'R3': {'C3': 3},
'R4': {'C1': 3},
'R5': {'C2': 3},
'R6': {'C3': 3},
'C1': {'R1': 3, 'R4': 3, 'E1': 4, 'C2': 8},
'C2': {'C1': 8, 'R2': 3, 'R5': 3, 'C3': 8},
'C3': {'C2': 8, 'R6': 3, 'R3': 3, 'E2': 4},
'E1': {'C1': 4},
'E2': {'C3': 4},
}

V_F = 1.4     #消防员健康状态下的速度为1.4m/s
# 理想状态下的基础通行时间 T_Base(i, j) (秒)
T_BASE_FORF = {}
for location, connections in DISTANCES.items():
    new_connections = {}
    for destination, distance in connections.items():
        new_distance = distance / V_F
        new_connections[destination] = new_distance
    T_BASE_FORF[location] = new_connections


class Constants:
    """定义模型中使用的所有常量、系数和初始状态。"""
    # 场景定义
    ROOM_NAMES = sorted([n for n, node in NODES.items() if node.type == 'Room'])
    ENTRY_NAMES = sorted([n for n, node in NODES.items() if node.type == 'Entry'])
    EXIT_NAMES = sorted([n for n, node in NODES.items() if node.type == 'Entry'])
    MAX_FIRE_FIGHTERS = 20  # 最大可出动的消防员数量 N_F^Max
    # 已知项
    N_R_INIT = {'R1': 9, 'R2': 3, 'R3': 4, 'R4': 6, 'R5': 2, 'R6': 5}   # 房间初始人数
    SEARCH_VOLUME_BASE = 48.0                                           # 基础搜救工作量
    SEARCH_RATE_BASE = 0.4                                              # 消防员基础搜救效率
    C_FIXED = 30                                                        # 消防员固定沉没成本
    P_Isdead = 4000.0                                                   # 死被困人人的成本
    P_Fisdead = 6000.0                                                  # 消防员死亡的成本
    R_ORIGIN = 'R2'                                                     # 火灾起始房间 
    H_F_INIT = 100.0                                                    # 消防员初始血量 
    P_F_INIT = 1.0                                                      # 消防员初始专业性 
    H_V_INIT = 100.0                                                    # 被困人员初始血量 
    V_P_INIT = 1.5                                                      # 被困人员初始速度 
    P_ESCAPE = 0.01                                                     # 被困人员基础逃跑概率
    GAMMA_HEALTH = 5.0                                                  # 健康对逃跑概率的影响因子
    # 权重
    WEIGHT_REVENUE = 10.0                                               # 总收益 R_Total 的权重
    WEIGHT_TIME_REWARD = 2.0                                            # 效率奖励 R_Time 的权重
    # 系数与阈值
    ALPHA_I = 0.05                                                      # 消防员血量衰减系数
    ALPHA_VD = 0.01                                                     # 被困人员血量衰减系数
    ALPHA_VS = 0.05                                                     # 烟雾对速度影响系数
    ALPHA_P = 0.005                                                     # 专业性随血量损失的衰减系数
    ALPHA_M = 0.05                                                      # 医疗成本系数
    ALPHA_RS = 0.4                                                      # 烟雾对搜救效率影响系数
    DT_SIM = 1.0                                                        # 仿真步长 (秒)
    # 火情/烟雾传播系数
    F_THRESHOLD = 2.0                                                   # 逃生阈值 (火情强度)
    K_MANI = 0.3                                                        # 火情距离衰减系数 
    K_SMOKE = 2                                                         # 烟雾距离衰减乘数 
    K_CORR = 0.8                                                        # 火情过渡修正系数 
    K_CORR_SMOKE = 0.6                                                  # 烟雾过渡修正系数 
    # 动态环境基线参数
    F0 = 1.0                                                            # 初始全局火情
    KF = 0.01                                                           # 火情指数增长率 
    S0 = 1.0                                                            # 初始全局烟雾
    KS = 0.05                                                           # 烟雾线性增长率 
    # 遗传算法参数
    POPULATION_SIZE = 50
    MAX_GENERATIONS = 100
    STOCHASTIC_RUNS = 10                                                # 每个基因的模拟次数（鲁棒性要求）
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.15
    TOURNAMENT_SIZE = 10
    STOP_CRITERIA_GEN = 8                                               # 适应度不再改善的代数阈值

class Firefighter:
    """消防员状态模型"""
    def __init__(self, id, entry_node, exit_node, init_H, init_P):
        self.id = id
        self.entry = entry_node
        self.exit = exit_node
        self.H_F = init_H
        self.P_F = init_P
        self.H_F_final = init_H
        self.is_active = True
        self.path_log = [entry_node]    # 记录路径
        self.rooms_searched = []        # 记录已搜救的房间
        self.time_spent = 0.0           # 记录总花费时间

class Victim:
    """被困人员状态模型"""
    def __init__(self, id, room_node, init_H, init_V):
        self.id = id
        self.room = room_node
        self.H_P = init_H
        self.V_P_init = init_V
        self.H_P_final = -1.0       # 默认未被发现
        self.is_found = False
        self.is_escaped = False     # 逃跑成功
        self.is_dead = False        # 死亡 (H_P <= 0)
        self.is_escaping = False    # 是否在逃亡的路途中

class SimulationCore:
    """实现数学模型中的所有动态方程和逻辑。"""
    def __init__(self, room_initial_victims):
        self.const = Constants
        self.R_ORIGIN = self.const.R_ORIGIN
        self.victims = self._init_victims(room_initial_victims)
        self.T_Total = 0.0      # 全局时间，每次模拟开始时重置
        self.final_metrics = {}
        self.firefighters = []
        self.F_states = {}

    def _init_victims(self, room_initial_victims):
        """初始化被困人员列表"""
        victims = []
        victim_id = 1
        for room, count in room_initial_victims.items():
            for _ in range(count):
                victims.append(Victim(victim_id, room, self.const.H_V_INIT, self.const.V_P_INIT))
                victim_id += 1
        return victims

    def _get_distance(self, i, j):
        """获取节点i到j的距离 D(i, j)"""
        return DISTANCES.get(i, {}).get(j, float('inf'))

    def _dijkstra_shortest_path(self, start_node, end_node):
        """
        使用 Dijkstra 算法查找最短路径（基于基础距离）。
        返回: 最短距离, 节点路径列表
        """
        if start_node == end_node:
            return 0.0, [start_node]

        distances = {node: float('inf') for node in NODES}
        distances[start_node] = 0
        previous_nodes = {node: None for node in NODES}
        priority_queue = [(0, start_node)]  # (distance, node)

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            if current_node == end_node:
                path = []
                temp_node = end_node
                while temp_node is not None:
                    path.insert(0, temp_node)
                    temp_node = previous_nodes[temp_node]
                return distances[end_node], path

            for neighbor, distance in DISTANCES.get(current_node, {}).items():
                new_distance = current_distance + distance
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        return float('inf'), [] # 无法到达

    def _calculate_global_fire_smoke(self, t):
        """计算全局火情基线 F_Global(t) 和全局烟雾基线 S_Global(t)"""
        F_Global = self.const.F0 * math.exp(self.const.KF * t)
        S_Global = self.const.S0 + self.const.KS * t
        return F_Global, S_Global

    def _calculate_environment(self, t, node_i):
        """计算节点 i 的实际火情 F_i(t) 和实际烟雾 S_i(t)"""
        if node_i not in NODES:
            return 0.0, 0.0

        F_Global, S_Global = self._calculate_global_fire_smoke(t)
        D_i_R_Origin, _ = self._dijkstra_shortest_path(node_i, self.R_ORIGIN)       # 查找节点到火源点的最短距离
        if D_i_R_Origin == float('inf'):
            D_i_R_Origin = 100.0

        node_type = NODES[node_i].type  # 节点类型修正项
        is_room = (node_type == 'Room')

        # 1. 实际火情
        delta_fire = 1.0 if is_room else self.const.K_CORR
        F_i = F_Global * math.exp(-self.const.K_MANI * D_i_R_Origin) * delta_fire

        # 2. 实际烟雾 
        delta_smoke = 1.0 if is_room else self.const.K_CORR_SMOKE
        distance_factor = self.const.K_SMOKE / max(1.0, D_i_R_Origin)
        S_i = S_Global * distance_factor * delta_smoke
        S_i = max(0.0, S_i)

        return F_i, S_i

    def _update_firefighter_state(self, F_k, node_i, dt):
        """更新消防员状态：血量和专业性"""
        if not F_k.is_active:
            return

        F_i, _ = self._calculate_environment(self.T_Total, node_i)# 确保环境计算使用全局时间 T_Total

        # 1. 消防员血量变化率 dH_Fk/dt
        dH_Fk_dt = -self.const.ALPHA_I * F_i
        F_k.H_F += dH_Fk_dt * dt
        F_k.H_F = max(0.0, F_k.H_F)

        # 2. 消防员专业性 P_Fk(t)
        F_k.P_F = self.const.P_F_INIT - self.const.ALPHA_P * (self.const.H_F_INIT - F_k.H_F)
        F_k.P_F = max(0.1, F_k.P_F) # 专业性最低为 0.1

        # 3. 检查状态
        if F_k.H_F <= 0.0:
            F_k.is_active = False

    def _update_victim_state(self, victim, current_time, dt):
        """更新被困人员状态：血量"""
        if victim.is_found or victim.is_escaped or victim.is_dead:
            return

        F_i, _ = self._calculate_environment(current_time, victim.room)

        if victim.H_P > 0.0:
            dH_Pi_dt = -self.const.ALPHA_VD * F_i
            victim.H_P += dH_Pi_dt * dt
            victim.H_P = max(0.0, victim.H_P)

        if victim.H_P <= 0.0:
            victim.is_dead = True

    def _calculate_escape_stochastic(self, victim, current_time, dt):
        """计算被困人员实际逃跑概率 P_Run(t) 并模拟逃跑。"""
        if victim.is_found or victim.is_escaped or victim.is_dead:
            return False

        F_i, _ = self._calculate_environment(current_time, victim.room)

        if F_i > self.const.F_THRESHOLD:
            health_ratio = victim.H_P / self.const.H_V_INIT
            P_Run = self.const.P_ESCAPE * (1.0 + self.const.GAMMA_HEALTH * (1.0 - health_ratio))
            P_Run = max(0.0, min(1.0, P_Run))

            if random.random() < P_Run * dt:
                victim.is_escaped = True
                return True
        return False

    def _calculate_objective(self, active_firefighters):
        """计算目标函数 Z = W_R * R_Total - C_Total + W_T * R_Time_Base"""
        
        R_Total = sum(v.H_P_final for v in self.victims if v.is_found and v.H_P_final > 0)# 确保只计算被发现的且血量大于0的个体

        N_F_active = len(active_firefighters)
        C_Fixed_Sunk = N_F_active * self.const.C_FIXED

        # 医疗成本
        C_Medical = sum((self.const.H_F_INIT - F_k.H_F_final) * self.const.ALPHA_M
                            for F_k in active_firefighters)
        # 统计死亡消防员
        N_F_dead = sum(1 for F_k in active_firefighters if F_k.H_F_final <= 0)
        C_F_Dead = N_F_dead * self.const.P_Fisdead # 新增：消防员死亡成本

        # 统计未救出人员
        # 死亡人数：H_P <= 0 或 is_dead 为 True
        N_Dead = sum(1 for v in self.victims if v.is_dead)
        N_Trapped = sum(1 for v in self.victims if not v.is_found and not v.is_escaped and not v.is_dead)
        C_Isdead = self.const.P_Isdead * (N_Dead + N_Trapped)

        #C_Purchase = N * self.const.P_Purchase

        C_Total = C_Fixed_Sunk + C_Medical + C_Isdead +C_F_Dead #+ C_Purchase

        # 效率奖励
        R_Time_Base = 100.0 / max(1.0, self.T_Total)

        # 目标函数 Z
        Z = (self.const.WEIGHT_REVENUE * R_Total) - C_Total + (self.const.WEIGHT_TIME_REWARD * R_Time_Base/100)

        # 记录最终指标
        self.final_metrics = {
            'R_Total': R_Total,
            'C_Total': C_Total,
            'R_Time_Base': R_Time_Base,
            'Z': Z,
            'C_Fixed_Sunk': C_Fixed_Sunk,
            'C_Medical': C_Medical,
            'C_Isdead': C_Isdead,
            'C_F_Dead': C_F_Dead,
            'N_Found': sum(1 for v in self.victims if v.is_found),
            'N_Escaped': sum(1 for v in self.victims if v.is_escaped),
            'N_Dead': sum(1 for v in self.victims if v.is_dead),
            'N_Trapped': sum(1 for v in self.victims if not v.is_found and not v.is_escaped and not v.is_dead), # 新增：困住但存活
            'H_F_Final': {f'{f.id}': f.H_F_final for f in active_firefighters},
            'T_Total': self.T_Total
        }

        return Z

    def _process_firefighter_step(self, F_k, F_state, dt):
        """在一个 dt 步长内，处理单个消防员的增量移动/搜救/状态切换。"""

        if F_state['is_searching']:
            room_node = F_state['current_node']
            _, S_i = self._calculate_environment(self.T_Total, room_node)
            P_Fk = F_k.P_F
            smoke_penalty = (1.0 - self.const.ALPHA_RS * S_i)
            R_Search = self.const.SEARCH_RATE_BASE * P_Fk * max(0.1, smoke_penalty)

            work_remaining = F_state['work_rem']

            if R_Search <= 0.0:
                work_done = 0.0
                dt_actual = dt
            else:
                work_possible = R_Search * dt
                if work_possible >= work_remaining:
                    dt_actual = work_remaining / R_Search  # 精确完成所需时间
                    work_done = work_remaining
                else:
                    dt_actual = dt
                    work_done = work_possible

            self._update_firefighter_state(F_k, room_node, dt_actual)
            F_state['work_rem'] -= work_done

            if F_state['work_rem'] <= 0.0:  # 搜救完成：切换到下一个任务（移动）
                F_state['is_searching'] = False         
                F_k.rooms_searched.append(room_node)

                for victim in self.victims: # 标记房间内所有存活者为已发现
                    if victim.room == room_node and not victim.is_found and not victim.is_dead:
                        victim.is_found = True
                        victim.H_P_final = victim.H_P

            return  # 本步长只执行搜救

        else:

            path = F_state['path_nodes']
            if F_state['path_idx'] >= len(path) - 1:
                F_state['is_completed'] = True# 已到达出口
                return

            prev_node = path[F_state['path_idx']]

            # 实时计算速度
            _, S_i = self._calculate_environment(self.T_Total, prev_node)
            V_F = 1.4   # V_F 基础速度
            speed_factor = F_k.P_F * (1.0 - self.const.ALPHA_VS * S_i)
            V_Actual = V_F * max(0.01, speed_factor)

            if V_Actual <= 0.01:# 速度极慢，无法移动，但必须持续扣血
                self._update_firefighter_state(F_k, prev_node, dt)
                return

            distance_remaining = F_state['dist_rem']
            distance_possible = V_Actual * dt

            if distance_possible >= distance_remaining:
                dt_actual = distance_remaining / V_Actual  # 精确完成移动所需时间
                distance_moved = distance_remaining
            else:
                dt_actual = dt
                distance_moved = distance_possible

            # 更新状态
            self._update_firefighter_state(F_k, prev_node, dt_actual)
            F_state['dist_rem'] -= distance_moved

            if F_state['dist_rem'] <= 0.0:# 移动完成：切换到下一个节点/任务
                F_state['path_idx'] += 1
                target_node = path[F_state['path_idx']]
                F_state['current_node'] = target_node
                F_k.path_log.append(target_node)  # 更新路径记录

                if NODES[target_node].type == 'Room':
                    F_state['is_searching'] = True
                    F_state['work_rem'] = self.const.SEARCH_VOLUME_BASE

                if F_state['path_idx'] < len(path) - 1:     # 准备下一段移动
                    next_next_node = path[F_state['path_idx'] + 1]
                    D_new = DISTANCES.get(target_node, {}).get(next_next_node, float('inf'))
                    F_state['dist_rem'] = D_new
                else:
                    F_state['is_completed'] = True

    def run_simulation(self, chromosome):
        """
        根据染色体执行一次完整的仿真。
        返回：目标函数值 Z
        """
        Z = -99999.0
        # 解码染色体并初始化消防员
        N_F, assignments, entries, exits = chromosome.decode()
        self.firefighters = []
        for i in range(N_F):
            F_id = f'F{i+1}'
            entry = self.const.ENTRY_NAMES[entries[i]]
            exit = self.const.EXIT_NAMES[exits[i]]
            self.firefighters.append(Firefighter(
                F_id,
                entry,
                exit,
                self.const.H_F_INIT,
                self.const.P_F_INIT
            ))

        # 构建每个消防员的路径
        firefighter_room_sequences = defaultdict(list)
        for room in assignments:
            # 确保分配的消防员ID在有效范围内 [1, N_F]
            if 1 <= assignments[room] <= N_F:
                F_idx = assignments[room] - 1
                F_name = f'F{F_idx + 1}'
                firefighter_room_sequences[F_name].append(room)

        # 使用 Dijkstra 构建完整的、可执行的路径
        final_routes = {} 

        for F_k in self.firefighters:
            F_id = F_k.id
            room_sequence = firefighter_room_sequences[F_id]

            # 路径顺序: Entry -> Rooms (顺序由 GA 决定) -> Exit
            current_path_sequence = [F_k.entry] + room_sequence + [F_k.exit]

            full_path_nodes = []

            for i in range(len(current_path_sequence) - 1):
                start = current_path_sequence[i]
                end = current_path_sequence[i + 1]

                dist, path = self._dijkstra_shortest_path(start, end)

                if dist == float('inf'):
                    full_path_nodes = []# 路径不可达，任务中断/失败
                    break

                    # 拼接路径，跳过起点以避免重复节点
                if full_path_nodes:
                    full_path_nodes.extend(path[1:])
                else:
                    full_path_nodes.extend(path)

            final_routes[F_id] = full_path_nodes  # 存储最终路径

        self.T_Total = 0.0

        self.F_states = {}
        active_firefighters = []  # 追踪所有未死亡/未完成任务的消防员

        for F_k in self.firefighters:
            route = final_routes.get(F_k.id, [])
            if len(route) < 2:
                continue  # 无效路径/未分配任务

            active_firefighters.append(F_k)    

            start_node = route[0]    # 初始化状态
            next_node = route[1]
            initial_dist, _ = self._dijkstra_shortest_path(start_node, next_node)
            initial_dist = DISTANCES.get(start_node, {}).get(next_node, initial_dist)  # 优先使用直接距离

            self.F_states[F_k.id] = {
                'current_node': start_node,
                'path_nodes': route,
                'path_idx': 0,
                'dist_rem': initial_dist,
                'is_searching': NODES[start_node].type == 'Room',
                'work_rem': self.const.SEARCH_VOLUME_BASE if NODES[start_node].type == 'Room' else 0.0,
                'is_completed': False,
                'time_spent': 0.0,
            }

        dt = self.const.DT_SIM


        while active_firefighters:
            T_Total_Start = self.T_Total

            # 消防员并行操作
            for F_k in active_firefighters[:]:  # 遍历副本以安全修改列表
                F_state = self.F_states[F_k.id]

                # 尝试执行 dt 时间步长的移动或搜锁

                # 假设我们有一个新函数处理步进
                self._process_firefighter_step(F_k, F_state, dt)

                # 检查状态
                if F_k.H_F <= 0.0:
                    F_k.is_active = False  # 消防员死亡

                if F_k.is_active == False or F_state['is_completed']:
                    active_firefighters.remove(F_k)
                    F_k.H_F_final = F_k.H_F
                    F_k.time_spent = self.T_Total  # 记录任务完成时间

            # 被困人员并行状态更新 (基于 T_Total_Start + dt)
            for victim in self.victims:
                if not victim.is_found and not victim.is_escaped and not victim.is_dead:
                    self._update_victim_state(victim, self.T_Total, dt)
                    self._calculate_escape_stochastic(victim, self.T_Total, dt)

            self.T_Total += dt


        # 计算目标函数 Z
        Z = self._calculate_objective(self.firefighters) # 保持不变

        return Z


class Chromosome:
    """染色体编码和解码"""
    def __init__(self, gene=None):
        self.const = Constants

        self.rooms_len = len(self.const.ROOM_NAMES)
        self.entry_exit_len = self.const.MAX_FIRE_FIGHTERS * 2
        self.total_len = 1 + self.rooms_len + self.entry_exit_len

        if gene is None:
            self.gene = self._initialize_random()
        else:
            self.gene = gene

        self.fitness_history = []
        self.avg_fitness = -float('inf')

    def _initialize_random(self):
        """随机初始化一个有效的染色体"""
        gene = [0] * self.total_len

        NF = random.randint(1, self.const.MAX_FIRE_FIGHTERS)
        gene[0] = NF

        for i in range(self.rooms_len):
            # 分配给 0 (不分配) 到 NF (出动的最大消防员ID) 之间
            gene[1 + i] = random.randint(1, NF)

        # 入口/出口 F_k
        for i in range(self.const.MAX_FIRE_FIGHTERS):
            gene[1 + self.rooms_len + i*2] = random.randint(0, len(self.const.ENTRY_NAMES) - 1)
            gene[1 + self.rooms_len + i*2 + 1] = random.randint(0, len(self.const.EXIT_NAMES) - 1)

        return gene

    def decode(self):
        """解码染色体为可执行的决策结构"""
        NF = self.gene[0]

        assignments = {}
        for i, room_name in enumerate(self.const.ROOM_NAMES):
            assignments[room_name] = self.gene[1 + i]

        entries = []
        exits = []
        # 只解码 NF 个消防员的入口和出口
        for i in range(NF):
            entries.append(self.gene[1 + self.rooms_len + i*2])
            exits.append(self.gene[1 + self.rooms_len + i*2 + 1])

        return NF, assignments, entries, exits

    def display_gene(self):
        """将染色体解析为可读的输出"""
        NF, assignments, entries, exits = self.decode()

        output = f"消防员人数 (N_F): {NF}\n"

        output += "房间分配 (Room -> Firefighter ID):\n"
        room_assignments = {r: f for r, f in assignments.items() if 0 < f <= NF}
        output += f"  {room_assignments}\n"

        F_paths = defaultdict(list)
        for room in self.const.ROOM_NAMES:
            F_id = assignments[room]
            if 0 < F_id <= NF:
                F_paths[F_id].append(room)

        output += "各消防员搜救房间顺序 (初始分配):\n"
        for i in range(NF):
            F_id = i + 1
            entry = self.const.ENTRY_NAMES[entries[i]]
            exit = self.const.EXIT_NAMES[exits[i]]
            path = " -> ".join(F_paths[F_id])
            if not path:
                path = "未分配任务"
            output += f"  F{F_id} (入口: {entry}, 出口: {exit}): {path}\n"

        return output

# 多进程工作函数
def worker_evaluate_fitness(chromosome_gene):
    """一个独立的、可被多进程调用的工作函数，执行鲁棒性评估。"""
    chromosome = Chromosome(gene=chromosome_gene)
    sim_runs = Constants.STOCHASTIC_RUNS

    fitness_list = []
    for _ in range(sim_runs):
        sim = SimulationCore(Constants.N_R_INIT)
        fitness = sim.run_simulation(chromosome)
        fitness_list.append(fitness)

    return np.mean(fitness_list)

class GeneticAlgorithm:
    """遗传算法主框架"""
    def __init__(self, sim_runs=10):
        self.const = Constants
        self.sim_runs = sim_runs
        self.population = []
        self.best_chromosome = None
        self.best_fitness_history = []

        self.max_workers = os.cpu_count()
        print(f"--- GA 初始化 --- (将使用 {self.max_workers} 个工作进程) ---")

    def initialize_population(self):
        """初始化种群"""
        self.population = [Chromosome() for _ in range(self.const.POPULATION_SIZE)]

    def calculate_fitness(self):
        """
        计算整个种群的适应度 (多进程修改版)
        """
        genes_to_evaluate = [chrom.gene for chrom in self.population]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(worker_evaluate_fitness, genes_to_evaluate))

        for i, avg_fitness in enumerate(results):
            self.population[i].avg_fitness = avg_fitness

        current_best = max(self.population, key=lambda x: x.avg_fitness)

        if self.best_chromosome is None or current_best.avg_fitness > self.best_chromosome.avg_fitness:
            self.best_chromosome = Chromosome(current_best.gene)
            self.best_chromosome.avg_fitness = current_best.avg_fitness

        self.best_fitness_history.append(self.best_chromosome.avg_fitness)

    def selection(self):
        """锦标赛选择"""
        selected = []
        for _ in range(self.const.POPULATION_SIZE):
            competitors = random.sample(self.population, self.const.TOURNAMENT_SIZE)
            winner = max(competitors, key=lambda x: x.avg_fitness)
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        """均匀交叉和两点交叉的混合"""
        p1_gene = parent1.gene[:]
        p2_gene = parent2.gene[:]

        rooms_len = parent1.rooms_len
        total_len = parent1.total_len
        room_start = 1
        room_end = 1 + rooms_len

        child1_gene = p1_gene[:]
        child2_gene = p2_gene[:]

        # 房间分配部分
        for i in range(rooms_len):
            if random.random() < 0.5:
                child1_gene[room_start + i] = p2_gene[room_start + i]
                child2_gene[room_start + i] = p1_gene[room_start + i]

        # NF 和 Entry/Exit 部分 
        point1 = random.randint(0, total_len - 1)
        point2 = random.randint(point1, total_len - 1)

        c1_nf_ee = p1_gene[0:point1] + p2_gene[point1:point2] + p1_gene[point2:]
        c2_nf_ee = p2_gene[0:point1] + p1_gene[point1:point2] + p2_gene[point2:]

        child1_gene[0] = c1_nf_ee[0]
        child1_gene[room_end:] = c1_nf_ee[room_end:]

        child2_gene[0] = c2_nf_ee[0]
        child2_gene[room_end:] = c2_nf_ee[room_end:]

        # 确保 NF 是有效的
        NF1 = max(1, min(self.const.MAX_FIRE_FIGHTERS, child1_gene[0]))     # 确保 NF 在有效范围内
        child1_gene[0] = NF1    # 更新 NF，以防其超出 MAX_FIRE_FIGHTERS
        for i in range(rooms_len):
            idx = room_start + i
            current_id = child1_gene[idx]

            # 如果分配 ID 不在 [1, NF1] 范围内（即 0 或 > NF1），则修正
            if not (1 <= current_id <= NF1):
                # 修正为 Child 1 的有效消防员 ID
                child1_gene[idx] = random.randint(1, NF1)


        NF2 = max(1, min(self.const.MAX_FIRE_FIGHTERS, child2_gene[0]))
        child2_gene[0] = NF2 # 更新 NF
        for i in range(rooms_len):
            idx = room_start + i
            current_id = child2_gene[idx]

            # 如果分配 ID 不在 [1, NF2] 范围内（即 0 或 > NF2），则修正
            if not (1 <= current_id <= NF2):
                # 修正为 Child 2 的有效消防员 ID
                child2_gene[idx] = random.randint(1, NF2)
        return Chromosome(child1_gene), Chromosome(child2_gene)

    def mutation(self, chromosome):
        """
        --- 改进 4: 变异操作确保房间分配约束 ---
        NF变异，房间分配变异，入口/出口变异。
        """
        gene = chromosome.gene[:]
        NF = gene[0]

        # NF 变异
        if random.random() < self.const.MUTATION_RATE:
            gene[0] = random.randint(1, self.const.MAX_FIRE_FIGHTERS)
            NF = gene[0] # 更新 NF

        # 房间分配变异
        if random.random() < self.const.MUTATION_RATE:
            room_idx = random.randint(1, chromosome.rooms_len)
            # 确保分配给的消防员ID在 [1, NF] 范围内 (强制分配给一个出动的消防员)
            if NF > 0:  
                gene[room_idx] = random.randint(1, NF)
            else:
                # 极端情况下NF=0，则强制分配给 F1 
                gene[room_idx] = 1

        # 入口/出口变异
        if random.random() < self.const.MUTATION_RATE:
            ee_idx = random.randint(chromosome.rooms_len + 1, chromosome.total_len - 1)
            gene[ee_idx] = random.randint(0, len(self.const.ENTRY_NAMES) - 1)

        return Chromosome(gene)

    def run(self):
        """运行遗传算法主循环"""
        self.initialize_population()
        self.calculate_fitness() # 计算初始种群适应度

        start_time = time.time()

        no_improvement_count = 0
        best_fitness_last = self.best_chromosome.avg_fitness

        print(f"--- 遗传算法开始 (使用 {self.max_workers} 个工作进程) ---")

        for generation in range(self.const.MAX_GENERATIONS):
            parents = self.selection()

            next_population = []
            # 精英保留
            next_population.append(self.best_chromosome)

            # 交叉和变异生成其余种群
            for i in range(0, self.const.POPULATION_SIZE - 1, 2):
                p1 = parents[i]
                p2 = parents[i+1] if i+1 < len(parents) else parents[i]

                if random.random() < self.const.CROSSOVER_RATE:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1, p2

                next_population.append(self.mutation(c1))
                if len(next_population) < self.const.POPULATION_SIZE:
                    next_population.append(self.mutation(c2))

            self.population = next_population[:self.const.POPULATION_SIZE]

            gen_start_time = time.time()
            self.calculate_fitness()
            gen_time = time.time() - gen_start_time

            current_best_fitness = self.best_chromosome.avg_fitness

            # 检查是否有改进
            if current_best_fitness <= best_fitness_last:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                best_fitness_last = current_best_fitness

            print(f"Gen {generation+1}/{self.const.MAX_GENERATIONS}: 最佳适应度: {current_best_fitness:.2f} (耗时: {gen_time:.2f}s), 未改善: {no_improvement_count}代")

            if no_improvement_count >= self.const.STOP_CRITERIA_GEN:
                print(f"\n达到停步机制 ({self.const.STOP_CRITERIA_GEN} 代未改善)。GA 停止。")
                break

        end_time = time.time()
        print(f"--- 遗传算法结束 (总耗时: {end_time - start_time:.2f} 秒) ---")

        return self.best_chromosome


def analyze_best_result(best_chromosome):
    """
    对最优基因进行详细的最终模拟，并输出结果。
    """
    const = Constants

    # 重新运行一次仿真以获取详细指标
    sim = SimulationCore(const.N_R_INIT)
    final_fitness = sim.run_simulation(best_chromosome)
    metrics = sim.final_metrics

    NF, assignments, entries, exits = best_chromosome.decode()

    found_per_room = defaultdict(int)
    for v in sim.victims:
        if v.is_found:
            found_per_room[v.room] += 1

    #  详细结果输出 

    print("\n" + "="*50)
    print("                最优决策方案详细分析")
    print("                (基于一次最终模拟)")
    print("="*50)

    # 基因适应度
    print(f"1. 鲁棒适应度 (多次模拟平均): {best_chromosome.avg_fitness:.2f}")
    print(f"   最终模拟适应度: {final_fitness:.2f}")

    # 决策变量解析
    print("\n2. 决策方案:")
    print(best_chromosome.display_gene())

    # 任务执行概况
    print("\n3. 任务结果和人员状态:")
    print(f"   参与任务的消防员人数: {NF} 人")
    print(f"   任务总耗时 (T_Total): {metrics['T_Total']:.2f} 秒")
    print(f"   被困人员总数: {len(sim.victims)} 人")
    print(f"   被发现人数: {metrics['N_Found']} 人")
    print(f"   逃跑成功人数 (未被消防员发现): {metrics['N_Escaped']} 人")
    print(f"   死亡人数 (H_P <= 0): {metrics['N_Dead']} 人")
    print(f"   困住但存活人数 (未发现且未逃跑): {metrics['N_Trapped']} ")

    # 房间搜救详情
    print("\n4. 各房间搜救详情:")
    for room in const.ROOM_NAMES:
        initial = const.N_R_INIT[room]
        found = found_per_room[room]
        escaped = sum(1 for v in sim.victims if v.room == room and v.is_escaped)
        dead = sum(1 for v in sim.victims if v.room == room and v.is_dead)
        trapped = sum(1 for v in sim.victims if v.room == room and not v.is_found and not v.is_escaped and not v.is_dead)
        print(f"   {room}: 初始人数 {initial}, 发现 {found}, 逃跑 {escaped}, 死亡 {dead}, 存活/困住 {trapped}")

    # 消防员状态
    print("\n5. 消防员最终血量和成本:")
    active_firefighters = [f for f in sim.firefighters if f.id in metrics['H_F_Final']]
    for F_k in active_firefighters:
        H_final = metrics['H_F_Final'].get(F_k.id, 0.0)
        H_loss = const.H_F_INIT - H_final
        print(f"   {F_k.id}: 最终血量 {H_final:.2f}, 血量损失 {H_loss:.2f}")

    # 目标函数解析
    print("\n6. 目标函数 (适应度) Z 解析:")
    print(f"   总收益 R_Total (被发现个体血量和): {metrics['R_Total']:.2f} (权重前)")
    print(f"   加权收益 W_R * R_Total: {(const.WEIGHT_REVENUE * metrics['R_Total']):.2f} (W_R={const.WEIGHT_REVENUE})")
    print(f"   总成本 C_Total (固定沉没 + 医疗): {metrics['C_Total']:.2f}")
    print(f"     - 固定沉没成本 C_Fixed^Sunk: {metrics['C_Fixed_Sunk']:.2f}")
    print(f"     - 医疗成本 C_Medical: {metrics['C_Medical']:.2f}")
    print(f"     - 被困人员死亡/困住成本 C_Isdead: {metrics['C_Isdead']:.2f}")
    print(f"     - 消防员死亡成本 C_F_Dead: {metrics['C_F_Dead']:.2f}")
    print(f"   效率奖励 R_Time (基础值): {metrics['R_Time_Base']:.2f} (T_Total={metrics['T_Total']:.2f})")
    print(f"   加权奖励 W_T * R_Time_Base: {(const.WEIGHT_TIME_REWARD * metrics['R_Time_Base']):.2f} (W_T={const.WEIGHT_TIME_REWARD})")
    print(f"   Z = (W_R * R_Total) - C_Total + (W_T * R_Time_Base) = {metrics['Z']:.2f}")
    print("="*50)

def main():
    """程序主入口点"""
    try:
        ga = GeneticAlgorithm(sim_runs=Constants.STOCHASTIC_RUNS)
        best_chromosome = ga.run()

        if best_chromosome:
            analyze_best_result(best_chromosome)
    except Exception as e:
        print(f"\n程序运行中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()