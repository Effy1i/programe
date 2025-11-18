import numpy as np
import random
from copy import deepcopy

class IntegratedSmokeGA:
    def __init__(self):
        # 初始化烟雾时间计算器
        self.smoke_calculator = SmokeBasedTimeCalculator()
        
        # 遗传算法参数
        self.E = ['E1', 'E2']  # 出入口
        self.R = [f'R{i}' for i in range(1, 7)]  # 房间
        self.NODES = self.E + self.R
        
        # 遗传算法运行参数
        self.ga_pop_size = 150
        self.ga_generations = 500
        self.ga_mutation_rate = 0.15
        self.ga_elite_size = 15
        
        # 存储计算出的时间矩阵（秒）
        self.TIME_MATRIX = None
        self.SEARCH_TIMES = None

    def set_environment(self, distance_matrix, smoke_levels):
        """设置环境参数：距离矩阵和烟雾浓度"""
        self.smoke_calculator.set_distance_matrix(distance_matrix)
        self.smoke_calculator.set_smoke_matrix(smoke_levels)
        
        # 重新计算时间矩阵（秒）
        self._update_time_matrices()
        
    def set_ga_parameters(self, pop_size=150, generations=500, mutation_rate=0.15, elite_size=15):
        """设置遗传算法参数"""
        self.ga_pop_size = pop_size
        self.ga_generations = generations
        self.ga_mutation_rate = mutation_rate
        self.ga_elite_size = elite_size

    def _update_time_matrices(self):
        """更新时间矩阵（秒）"""
        # 计算通行时间矩阵（秒）
        self.TIME_MATRIX = self.smoke_calculator.generate_travel_time_matrix()
        
        # 计算并存储搜查时间（秒）
        self.SEARCH_TIMES = self.smoke_calculator.generate_search_time_dict()

    def get_time(self, node1, node2):
        """根据节点名获取通行时间 T(i, j) - 秒"""
        i = self.NODES.index(node1)
        j = self.NODES.index(node2)
        return self.TIME_MATRIX[i, j]

    def get_total_room_cost(self, room):
        """获取房间的固定搜查和清扫时间 S_i + C - 秒"""
        return self.SEARCH_TIMES.get(room, 0) + self.smoke_calculator.cleanup_time

    def calculate_path_time(self, path):
        """计算给定路径的总耗时 - 秒"""
        if not path or len(path) < 2:
            return 0
        
        # 1. 计算通行时间
        total_travel_time = sum(self.get_time(path[i], path[i+1]) for i in range(len(path) - 1))
        
        # 2. 计算房间操作时间 (搜查 + 清扫)
        rooms_in_path = [node for node in path if node in self.R]
        total_room_time = sum(self.get_total_room_cost(room) for room in rooms_in_path)
        
        return total_travel_time + total_room_time

    def decode_and_evaluate(self, chromosome):
        """评估染色体适应度"""
        room_genes = chromosome[:6]
        end_genes = chromosome[6:]
        
        end_f1 = self.E[end_genes[0] - 1] 
        end_f2 = self.E[end_genes[1] - 1]

        path_seq_f1 = []
        path_seq_f2 = []
        
        for room, fk in room_genes:
            if fk == 'F1':
                path_seq_f1.append(room)
            else:
                path_seq_f2.append(room)

        time_f1 = time_f2 = 0
        
        if path_seq_f1:
            start_node = 'E1'
            full_path_f1 = [start_node] + path_seq_f1 + [end_f1]
            time_f1 = self.calculate_path_time(full_path_f1)
        
        if path_seq_f2:
            start_node = 'E2'
            full_path_f2 = [start_node] + path_seq_f2 + [end_f2]
            time_f2 = self.calculate_path_time(full_path_f2)

        T_max = max(time_f1, time_f2)
        fitness = 1 / (T_max + 1e-6)

        return fitness, T_max, time_f1, time_f2

    def initialize_population(self, pop_size):
        """初始化种群"""
        population = []
        for _ in range(pop_size):
            assigned_rooms = [(r, random.choice(['F1', 'F2'])) for r in self.R]
            random.shuffle(assigned_rooms)
            end_genes = [random.randint(1, 2), random.randint(1, 2)]
            chromosome = assigned_rooms + end_genes
            population.append(chromosome)
        return population

    def crossover(self, parent1, parent2):
        """交叉操作"""
        room_genes_size = 6
        p1 = deepcopy(parent1)
        p2 = deepcopy(parent2)
        
        p1_rooms = p1[:room_genes_size]
        p2_rooms = p2[:room_genes_size]
        
        start, end = sorted(random.sample(range(room_genes_size), 2))
        
        child_rooms = [None] * room_genes_size
        child_rooms[start:end] = p1_rooms[start:end]
        
        p1_rooms_set = {room for room, fk in p1_rooms[start:end]}
        p2_sequence = [item for item in p2_rooms if item[0] not in p1_rooms_set]
        
        fill_index = end
        for item in p2_sequence:
            if fill_index >= room_genes_size:
                fill_index = 0
            while child_rooms[fill_index] is not None:
                fill_index += 1
                if fill_index >= room_genes_size:
                    fill_index = 0
            child_rooms[fill_index] = item
            fill_index += 1
        
        p1_ends = p1[6:]
        p2_ends = p2[6:]
        child_ends = [p1_ends[0], p2_ends[1]]

        return child_rooms + child_ends

    def mutate(self, chromosome, mutation_rate):
        """变异操作"""
        mutated_chromosome = deepcopy(chromosome)
        room_genes_size = 6
        
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(room_genes_size), 2)
            mutated_chromosome[idx1], mutated_chromosome[idx2] = \
                mutated_chromosome[idx2], mutated_chromosome[idx1]

        if random.random() < mutation_rate:
            idx = random.randint(0, room_genes_size - 1)
            room, fk = mutated_chromosome[idx]
            new_fk = 'F2' if fk == 'F1' else 'F1'
            mutated_chromosome[idx] = (room, new_fk)
            
        if random.random() < mutation_rate:
            current_end = mutated_chromosome[6]
            mutated_chromosome[6] = 3 - current_end 
            
        if random.random() < mutation_rate:
            current_end = mutated_chromosome[7]
            mutated_chromosome[7] = 3 - current_end 
            
        return mutated_chromosome

    def run_optimization(self):
        """运行完整的优化过程"""
        if self.TIME_MATRIX is None:
            raise ValueError("请先设置环境参数")
        
        population = self.initialize_population(self.ga_pop_size)
        best_fitness = 0
        best_chromosome = None
        
        print(f"--- 遗传算法参数: 种群={self.ga_pop_size}, 代数={self.ga_generations}, 变异率={self.ga_mutation_rate} ---")
        
        for generation in range(self.ga_generations):
            results = [(chromosome, *self.decode_and_evaluate(chromosome)) for chromosome in population]
            results.sort(key=lambda x: x[1], reverse=True)
            
            current_best = results[0]
            
            if current_best[1] > best_fitness:
                best_fitness = current_best[1]
                best_chromosome = current_best[0]
                
            if generation % 50 == 0 or generation == self.ga_generations - 1:
                T_max = 1 / best_fitness
                _, _, T1_best_print, T2_best_print = self.decode_and_evaluate(best_chromosome)
                print(f"Gen {generation:03d}: Best Max Time = {T_max:.1f} 秒 (F1={T1_best_print:.1f}, F2={T2_best_print:.1f})")
                
            new_population = [r[0] for r in results[:self.ga_elite_size]]
            
            total_fitness = sum(r[1] for r in results)
            selection_probabilities = [r[1] / total_fitness for r in results]
            
            while len(new_population) < self.ga_pop_size:
                parents = random.choices(results, weights=selection_probabilities, k=2)
                parent1 = parents[0][0]
                parent2 = parents[1][0]
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, self.ga_mutation_rate)
                new_population.append(child)
                
            population = new_population

        final_fitness, final_t_max, t1, t2 = self.decode_and_evaluate(best_chromosome)
        
        return final_t_max, best_chromosome, t1, t2

    def seconds_to_minutes(self, seconds):
        """将秒转换为分钟（用于显示）"""
        return seconds / 60.0

    def display_results(self, final_T_max, final_chromosome, T1, T2):
        """显示优化结果"""
        room_genes = final_chromosome[:6]
        end_genes = final_chromosome[6:]

        path_f1_rooms = [r for r, fk in room_genes if fk == 'F1']
        path_f2_rooms = [r for r, fk in room_genes if fk == 'F2']

        final_end_f1 = self.E[end_genes[0] - 1]
        final_end_f2 = self.E[end_genes[1] - 1]

        def get_final_path_str(start, room_seq, end):
            return f"{start} -> {' -> '.join(room_seq)} -> {end}"

        print("\n### 遗传算法最终结果 ###")
        print(f"F1 房间分配和访问顺序: {path_f1_rooms}")
        print(f"F2 房间分配和访问顺序: {path_f2_rooms}")
        print(f"F1 终点选择: {final_end_f1}")
        print(f"F2 终点选择: {final_end_f2}")
        
        print(f"F1 最终路径: {get_final_path_str('E1', path_f1_rooms, final_end_f1)}")
        print(f"   耗时: {T1:.1f} 秒 ({self.seconds_to_minutes(T1):.2f} 分钟)")
        print(f"F2 最终路径: {get_final_path_str('E2', path_f2_rooms, final_end_f2)}")
        print(f"   耗时: {T2:.1f} 秒 ({self.seconds_to_minutes(T2):.2f} 分钟)")

        print(f"\n**🔥 最终最小化最长耗时 (MAX Time):**")
        print(f"    **{final_T_max:.1f} 秒 ({self.seconds_to_minutes(final_T_max):.2f} 分钟)**")
        print("-" * 50)

    def display_current_times(self):
        """显示当前计算的时间矩阵"""
        if self.TIME_MATRIX is not None:
            print("\n--- 当前通行时间矩阵 (秒) ---")
            print("      " + "   ".join(f"{node:>3}" for node in self.NODES))
            for i, node in enumerate(self.NODES):
                row = " ".join(f"{self.TIME_MATRIX[i, j]:6.1f}" for j in range(len(self.NODES)))
                print(f"{node:3}  {row}")
            
            print("\n--- 房间操作时间 (秒) ---")
            for room in self.R:
                if room in self.SEARCH_TIMES:
                    search_time = self.SEARCH_TIMES[room]
                    total_time = self.get_total_room_cost(room)
                    print(f"{room}: 搜查 {search_time:.1f}秒 + 清扫 {self.smoke_calculator.cleanup_time}秒 = 总计 {total_time:.1f}秒")


# 烟雾时间计算器类（使用秒）
class SmokeBasedTimeCalculator:
    def __init__(self):
        # 默认参数（全部使用米和秒）
        self.normal_walking_speed = 1.4  # 米/秒
        self.inspection_rate = 0.3497  # 平方米/秒 (48平方米/137.2秒 ≈ 0.3497 m²/s)
        self.alpha = 0.121  # 烟雾对移动的影响系数
        self.beta = 0.121  # 烟雾对检查的影响系数
        self.typical_office_area = 48.0  # 平方米
        self.cleanup_time = 10  # 清扫时间：10秒

        # 节点定义
        self.nodes = ['E1', 'E2', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
        self.rooms = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
        self.exits = ['E1', 'E2']

        # 初始化距离矩阵和烟雾浓度矩阵
        self.distance_matrix = None
        self.smoke_levels = None

    def set_distance_matrix(self, distance_matrix):
        """设置距离矩阵（单位：米）"""
        self.distance_matrix = np.array(distance_matrix)
        print("距离矩阵设置成功！")

    def set_smoke_levels(self, smoke_dict):
        """通过字典设置每个节点的烟雾浓度"""
        self.smoke_levels = smoke_dict
        print("烟雾浓度设置成功！")

    def set_smoke_matrix(self, smoke_matrix):
        """通过矩阵设置烟雾浓度"""
        if len(smoke_matrix) != len(self.nodes):
            print(f"错误：烟雾矩阵大小应为 {len(self.nodes)}")
            return

        self.smoke_levels = {}
        for i, node in enumerate(self.nodes):
            self.smoke_levels[node] = smoke_matrix[i]
        print("烟雾浓度矩阵设置成功！")

    def calculate_travel_time_between_nodes(self, node_i, node_j):
        """计算两个节点之间的通行时间（单位：秒）"""
        if self.distance_matrix is None or self.smoke_levels is None:
            print("错误：请先设置距离矩阵和烟雾浓度")
            return None

        try:
            i = self.nodes.index(node_i)
            j = self.nodes.index(node_j)

            distance = self.distance_matrix[i, j]  # 米
            smoke_i = self.smoke_levels[node_i]
            smoke_j = self.smoke_levels[node_j]

            avg_smoke = (smoke_i + smoke_j) / 2
            base_travel_time = distance / self.normal_walking_speed
            travel_time = base_travel_time * (1 + self.alpha * avg_smoke)

            return travel_time

        except ValueError:
            print(f"错误：节点 {node_i} 或 {node_j} 不存在")
            return None

    def calculate_search_time(self, room):
        """计算指定房间的搜查时间（单位：秒）"""
        if self.smoke_levels is None:
            print("错误：请先设置烟雾浓度")
            return None

        if room not in self.smoke_levels:
            print(f"错误：房间 {room} 的烟雾浓度未设置")
            return None

        smoke_level = self.smoke_levels[room]
        base_search_time = self.typical_office_area / self.inspection_rate
        search_time = base_search_time * (1 + self.beta * smoke_level)

        return search_time

    def get_total_room_operation_time(self, room):
        """获取房间的总操作时间：搜查 + 清扫（单位：秒）"""
        search_time = self.calculate_search_time(room)
        if search_time is None:
            return None
        return search_time + self.cleanup_time

    def generate_travel_time_matrix(self):
        """生成完整的通行时间矩阵（单位：秒）"""
        if self.distance_matrix is None or self.smoke_levels is None:
            print("错误：请先设置距离矩阵和烟雾浓度")
            return None

        n = len(self.nodes)
        travel_time_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                node_i = self.nodes[i]
                node_j = self.nodes[j]
                travel_time_matrix[i, j] = self.calculate_travel_time_between_nodes(node_i, node_j)

        return travel_time_matrix

    def generate_search_time_dict(self):
        """生成所有房间的搜查时间字典（单位：秒）"""
        if self.smoke_levels is None:
            print("错误：请先设置烟雾浓度")
            return None

        search_times = {}
        for room in self.rooms:
            if room in self.smoke_levels:
                search_times[room] = self.calculate_search_time(room)

        return search_times


# 使用示例
if __name__ == "__main__":
    # 创建整合系统
    system = IntegratedSmokeGA()
    
    # 设置环境参数
    example_distance_matrix = [
        [0, 24, 7, 15, 23, 7, 15, 23],  # E1
        [24, 0, 23, 15, 7, 23, 15, 7],   # E2
        [7, 23, 0, 8, 16, 6, 14, 22],   # R1
        [15, 15, 8, 0, 8, 14, 6, 14],    # R2
        [23, 7, 16, 8, 0, 22, 14, 6],     # R3
        [7, 23, 6, 14, 22, 0, 8, 16],     # R4
        [15, 15, 14, 6, 14, 8, 0, 8],    # R5
        [23, 7, 22, 14, 6, 16, 8, 0]     # R6
    ]

    example_smoke_levels = [1, 0.4, 0.125, 0.0625, 0.0416, 0.125, 0.0625, 0.0416]
    
    system.set_environment(example_distance_matrix, example_smoke_levels)
    
    # 显示当前计算的时间
    system.display_current_times()
    
    # 运行优化
    final_T_max, final_chromosome, T1, T2 = system.run_optimization()
    
    # 显示结果
    system.display_results(final_T_max, final_chromosome, T1, T2)