# ============================================================
# GENETIC ALGORITHM - PHÂN BỔ CÔNG SUẤT 
# ============================================================
# BÀI TOÁN: 10 trạm phát (AP), 5 người dùng (UE)
# MỤC TIÊU: Tìm cách phân bổ công suất để tốc độ truyền cao nhất
# RÀNG BUỘC: Mỗi AP không vượt quá 100mW
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Cấu hình font
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

# ============================================
# THAM SỐ HỆ THỐNG
# ============================================
M = 10              # Số trạm phát (Access Points)
K = 5               # Số người dùng (User Equipments) 
P_MAX = 100.0       # Công suất tối đa mỗi AP (mW)
SIGMA2 = 1.0        # Nhiễu nền (mW)
AREA_SIZE = 1000    # Khu vực (m x m)

np.random.seed(42)  # Để kết quả giống nhau mỗi lần chạy

# ============================================
# KHỞI TẠO HỆ THỐNG
# ============================================
def initialize_system():
    """Tạo hệ thống ngẫu nhiên: vị trí APs, UEs và hệ số kênh truyền"""
    # Vị trí ngẫu nhiên
    ap_positions = np.random.uniform(0, AREA_SIZE, (M, 2))
    ue_positions = np.random.uniform(0, AREA_SIZE, (K, 2))
    
    # Tính khoảng cách
    distances = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            distances[m, k] = np.linalg.norm(ap_positions[m] - ue_positions[k])
    
    # Hệ số kênh truyền (càng gần càng tốt)
    path_loss = 1.0 / (1.0 + (distances / 100.0) ** 2)
    shadow_fading = 10 ** (np.random.normal(0, 8, (M, K)) / 10)
    beta = path_loss * shadow_fading
    
    return beta, ap_positions, ue_positions

# Tạo hệ thống
beta, ap_pos, ue_pos = initialize_system()

print("="*60)
print("   HỆ THỐNG CELL-FREE MASSIVE MIMO")
print("="*60)
print(f"Số trạm phát (M): {M}")
print(f"Số người dùng (K): {K}")
print(f"Công suất tối đa mỗi trạm: {P_MAX} mW")
print("="*60)

# ============================================
# HÀM MỤC TIÊU: SUM-RATE
# ============================================
def calculate_sum_rate(power_vector):
    """
    Tính tổng tốc độ truyền của toàn bộ hệ thống
    Input: power_vector - vector công suất [M*K]
    Output: Sum-Rate (bits/s/Hz) - càng cao càng tốt
    """
    # Chuyển vector thành ma trận [M, K]
    P = power_vector.reshape(M, K)
    
    sum_rate = 0.0
    for k in range(K):  # Với mỗi người dùng k
        # Tín hiệu mong muốn (từ tất cả các AP)
        signal = np.sum(np.sqrt(P[:, k]) * np.sqrt(beta[:, k])) ** 2
        
        # Can nhiễu (từ người dùng khác)
        interference = 0.0
        for j in range(K):
            if j != k:
                interference += np.sum(np.sqrt(P[:, j]) * np.sqrt(beta[:, k])) ** 2
        
        # SINR và Rate
        sinr = signal / (interference + SIGMA2)
        rate = np.log2(1 + sinr)
        sum_rate += rate
    
    return sum_rate

def fitness_function(x):
    """Hàm fitness cho GA (minimize)"""
    return -calculate_sum_rate(x)

# ============================================
# GENETIC ALGORITHM  
# ============================================
class GeneticAlgorithm:
    """
    Thuật toán di truyền để tìm phân bổ công suất tối ưu
    
    Hoạt động như tiến hóa tự nhiên:
    1. Tạo quần thể nghiệm ngẫu nhiên (50 cá thể)
    2. Chọn nghiệm tốt làm "cha mẹ" 
    3. "Lai tạo" cha mẹ tạo ra "con cái"
    4. "Đột biến" con cái một chút
    5. Lặp lại 100 thế hệ
    """
    
    def __init__(self, pop_size=50, max_gen=100, pc=0.8, pm=0.2):
        self.pop_size = pop_size    # Kích thước quần thể
        self.n_vars = M * K         # Số biến (50 = 10×5)
        self.max_gen = max_gen      # Số thế hệ
        self.pc = pc               # Xác suất lai ghép
        self.pm = pm               # Xác suất đột biến
        
        # Lưu lịch sử để vẽ biểu đồ
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_population(self):
        """Tạo 50 nghiệm ngẫu nhiên"""
        return np.random.uniform(0, P_MAX/K, (self.pop_size, self.n_vars))
    
    def tournament_selection(self, pop, fitness_values, k=3):
        """
        Chọn lọc kiểu đấu trường:
        - Chọn 3 cá thể ngẫu nhiên
        - Lấy cá thể tốt nhất
        """
        indices = np.random.randint(0, len(pop), k)
        best_idx = indices[np.argmin(fitness_values[indices])]
        return pop[best_idx].copy()
    
    def arithmetic_crossover(self, parent1, parent2):
        """
        Lai ghép số học:
        - Con1 = α×Cha1 + (1-α)×Cha2
        - Con2 = α×Cha2 + (1-α)×Cha1
        """
        if np.random.rand() < self.pc:
            alpha = np.random.rand()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def gaussian_mutation(self, individual):
        """
        Đột biến Gaussian:
        - Thêm nhiễu ngẫu nhiên vào một số gen
        """
        if np.random.rand() < self.pm:
            noise = np.random.randn(len(individual)) * (P_MAX / K) * 0.1
            mask = np.random.rand(len(individual)) < 0.1
            individual[mask] += noise[mask]
            individual = np.clip(individual, 0, P_MAX/K)
        return individual
    
    def repair_solution(self, individual):
        """
        Sửa nghiệm vi phạm ràng buộc:
        - Nếu tổng công suất AP > P_MAX → chia tỷ lệ
        """
        P = individual.reshape(M, K)
        for m in range(M):
            total_power = np.sum(P[m, :])
            if total_power > P_MAX:
                P[m, :] = P[m, :] * (P_MAX / total_power)
        return P.flatten()
    
    def run(self, verbose=True):
        """Chạy thuật toán GA"""
        
        if verbose:
            print("\n" + "="*60)
            print("   BẮT ĐẦU GENETIC ALGORITHM")
            print("="*60)
            print(f"Kích thước quần thể: {self.pop_size}")
            print(f"Số thế hệ: {self.max_gen}")
            print("-"*60)
        
        # Khởi tạo quần thể
        pop = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')
        
        # Vòng lặp tiến hóa
        for gen in range(self.max_gen):
            # Sửa lỗi ràng buộc
            pop = np.array([self.repair_solution(ind) for ind in pop])
            
            # Đánh giá fitness
            fitness_values = np.array([fitness_function(ind) for ind in pop])
            
            # Cập nhật nghiệm tốt nhất
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_fitness:
                best_fitness = fitness_values[min_idx]
                best_solution = pop[min_idx].copy()
            
            # Lưu lịch sử
            self.best_fitness_history.append(-best_fitness)
            self.avg_fitness_history.append(-np.mean(fitness_values))
            
            # In tiến trình
            if verbose and (gen % 20 == 0 or gen == self.max_gen - 1):
                print(f"Thế hệ {gen:3d}: Sum-Rate = {-best_fitness:.4f} bits/s/Hz")
            
            # Tạo thế hệ mới
            new_pop = []
            
            # Elitism: Giữ 10% tốt nhất
            elite_count = int(0.1 * self.pop_size)
            elite_indices = np.argsort(fitness_values)[:elite_count]
            for idx in elite_indices:
                new_pop.append(pop[idx].copy())
            
            # Sinh sản phần còn lại
            while len(new_pop) < self.pop_size:
                # Chọn cha mẹ
                p1 = self.tournament_selection(pop, fitness_values)
                p2 = self.tournament_selection(pop, fitness_values)
                
                # Lai ghép
                c1, c2 = self.arithmetic_crossover(p1, p2)
                
                # Đột biến
                c1 = self.gaussian_mutation(c1)
                c2 = self.gaussian_mutation(c2)
                
                new_pop.extend([c1, c2])
            
            pop = np.array(new_pop[:self.pop_size])
        
        if verbose:
            print("="*60)
            print("KẾT THÚC THUẬT TOÁN")
            print(f"Nghiệm tốt nhất: {-best_fitness:.4f} bits/s/Hz")
            print("="*60)
        
        return best_solution, -best_fitness, {
            'best_history': self.best_fitness_history,
            'avg_history': self.avg_fitness_history
        }

# ============================================
# PHƯƠNG PHÁP THAM CHIẾU (BASELINE)
# ============================================
def equal_power_allocation():
    """Phân bổ đều: mỗi AP chia đều cho 5 UE"""
    P_equal = np.ones((M, K)) * (P_MAX / K)
    sum_rate_equal = calculate_sum_rate(P_equal.flatten())
    return P_equal.flatten(), sum_rate_equal

# ============================================
# VẼ BIỂU ĐỒ
# ============================================
def plot_convergence(stats, baseline_rate):
    """Vẽ quá trình hội tụ"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = range(len(stats['best_history']))
    ax.plot(generations, stats['best_history'], 'r-', linewidth=2.5, label='GA (Tốt nhất)')
    ax.plot(generations, stats['avg_history'], 'b--', linewidth=2, label='GA (Trung bình)')
    ax.axhline(y=baseline_rate, color='gray', linestyle=':', linewidth=2, label=f'Baseline: {baseline_rate:.3f}')
    
    ax.set_xlabel('Thế hệ', fontsize=14)
    ax.set_ylabel('Sum-Rate (bits/s/Hz)', fontsize=14)
    ax.set_title('Quá trình hội tụ của Genetic Algorithm', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=300, bbox_inches='tight')
    print("Đã lưu: convergence.png")

def plot_heatmap(power_vector):
    """Vẽ bản đồ phân bổ công suất"""
    P = power_vector.reshape(M, K)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(P, cmap='YlOrRd', aspect='auto')
    
    # Hiển thị giá trị
    for m in range(M):
        for k in range(K):
            ax.text(k, m, f'{P[m,k]:.1f}', ha='center', va='center', 
                   color='black', fontweight='bold')
    
    ax.set_xlabel('User Equipment (UE)', fontsize=13)
    ax.set_ylabel('Access Point (AP)', fontsize=13)
    ax.set_title('Bản đồ phân bổ công suất (mW)', fontsize=15)
    
    plt.colorbar(im, ax=ax, label='Công suất (mW)')
    plt.tight_layout()
    plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
    print("Đã lưu: heatmap.png")

def plot_comparison(ga_rate, baseline_rate):
    """Vẽ so sánh GA vs Baseline"""
    methods = ['Baseline\n(Phân bổ đều)', 'Genetic Algorithm\n(Tối ưu hóa)']
    rates = [baseline_rate, ga_rate]
    colors = ['#ff7f7f', '#87ceeb']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, rates, color=colors, alpha=0.8, edgecolor='black')
    
    # Thêm giá trị
    for rate, bar in zip(rates, bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.4f}\nbits/s/Hz', ha='center', va='bottom', fontsize=12)
    
    # Cải thiện %
    improvement = (ga_rate - baseline_rate) / baseline_rate * 100
    ax.text(0.5, max(rates) * 0.5, f'Cải thiện: +{improvement:.1f}%', 
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_ylabel('Sum-Rate (bits/s/Hz)', fontsize=14)
    ax.set_title('So sánh hiệu năng: GA vs Baseline', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
    print("Đã lưu: comparison.png")

# ============================================
# CHẠY THỬ NGHIỆM
# ============================================
if __name__ == "__main__":
    # Tính Baseline
    print("\nTính toán phương pháp phân bổ đều...")
    baseline_solution, baseline_rate = equal_power_allocation()
    print(f"Baseline Sum-Rate: {baseline_rate:.4f} bits/s/Hz")
    
    # Chạy GA
    ga = GeneticAlgorithm(pop_size=50, max_gen=100, pc=0.8, pm=0.2)
    best_solution, best_rate, stats = ga.run()
    
    # So sánh kết quả
    print("\n" + "="*60)
    print("   KẾT QUẢ CUỐI CÙNG")
    print("="*60)
    print(f"Baseline (Phân bổ đều):     {baseline_rate:.4f} bits/s/Hz")
    print(f"Genetic Algorithm:          {best_rate:.4f} bits/s/Hz")
    improvement = (best_rate - baseline_rate) / baseline_rate * 100
    print(f"Cải thiện:                  +{improvement:.2f}%")
    print("="*60)
    
    # Tạo biểu đồ
    print("\nĐang tạo biểu đồ...")
    plot_convergence(stats, baseline_rate)
    plot_heatmap(best_solution)
    plot_comparison(best_rate, baseline_rate)
    
    print("\nHoàn thành! Đã tạo 3 file ảnh cho báo cáo:")
    print("- convergence.png: Quá trình hội tụ")
    print("- heatmap.png: Phân bổ công suất tối ưu") 
    print("- comparison.png: So sánh GA vs Baseline")
