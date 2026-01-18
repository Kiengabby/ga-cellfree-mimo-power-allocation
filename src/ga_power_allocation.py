# ============================================================
# GENETIC ALGORITHM - PHÂN BỔ CÔNG SUẤT CHO CELL-FREE MASSIVE MIMO
# ============================================================
# 
# BÀI TOÁN: 
# - Có 10 trạm phát (AP), 5 người dùng (UE)
# - Tìm công suất tối ưu p_mk (AP m phát cho UE k) 
# - Mục tiêu: Tối đa tổng tốc độ truyền (Sum-Rate)
# - Ràng buộc: Tổng công suất mỗi AP ≤ 100mW
#
# GENETIC ALGORITHM:
# 1. Tạo 50 nghiệm ngẫu nhiên (quần thể)
# 2. Lặp 100 thế hệ: Chọn lọc → Lai ghép → Đột biến  
# 3. Nghiệm tốt nhất = phân bổ công suất tối ưu
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Cấu hình font tiếng Việt cho biểu đồ
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

# ============================================
# 1. THAM SỐ HỆ THỐNG (THEO BÁO CÁO)
# ============================================
M = 10              # Số Access Points (APs) - Trạm phát sóng
K = 5               # Số User Equipments (UEs) - Người dùng  
P_MAX = 100.0       # Công suất tối đa mỗi AP (mW) - Giới hạn phần cứng
SIGMA2 = 1.0        # Công suất nhiễu (mW) - Nhiễu nền của hệ thống
AREA_SIZE = 1000    # Kích thước vùng phủ sóng (m x m) - Khu vực hoạt động

# Random seed để tái tạo kết quả giống nhau mỗi lần chạy
np.random.seed(42)

# ============================================
# 2. MÔ HÌNH HỆ THỐNG
# ============================================
def initialize_system():
    """
    Khởi tạo hệ thống Cell-Free Massive MIMO:
    
    GIẢI THÍCH CƠ BẢN:
    - Cell-Free nghĩa là không có ranh giới cell, tất cả AP phối hợp
    - Mỗi AP có thể phục vụ đồng thời nhiều UE
    - Chất lượng kênh truyền phụ thuộc khoảng cách và điều kiện môi trường
    
    TRẢ VỀ:
    - beta: Ma trận hệ số kênh truyền [M×K] (Large-scale fading)
    - ap_positions: Vị trí các AP [M×2] 
    - ue_positions: Vị trí các UE [K×2]
    """
    print("🔧 Đang khởi tạo hệ thống...")
    
    # Đặt APs và UEs ngẫu nhiên trong khu vực
    ap_positions = np.random.uniform(0, AREA_SIZE, (M, 2))
    ue_positions = np.random.uniform(0, AREA_SIZE, (K, 2))
    
    # Tính khoảng cách giữa mỗi AP và mỗi UE
    distances = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            # Khoảng cách Euclidean giữa AP m và UE k
            distances[m, k] = np.linalg.norm(ap_positions[m] - ue_positions[k])
    
    # Mô hình Large-scale fading (suy hao theo khoảng cách)
    # Công thức: β_mk = path_loss × shadow_fading
    
    # 1. Path loss: suy hao do khoảng cách (càng xa càng yếu)
    path_loss = 1.0 / (1.0 + (distances / 100.0) ** 2)
    
    # 2. Shadow fading: suy hao do vật cản (tường, cây, nhà...)
    # Phân phối log-normal với độ lệch chuẩn 8 dB
    shadow_fading = 10 ** (np.random.normal(0, 8, (M, K)) / 10)
    
    # Hệ số kênh truyền cuối cùng
    beta = path_loss * shadow_fading
    
    print(f"✅ Đã tạo {M} APs và {K} UEs")
    print(f"✅ Tính toán ma trận kênh truyền β [{M}×{K}]")
    
    return beta, ap_positions, ue_positions

# Khởi tạo hệ thống
beta, ap_pos, ue_pos = initialize_system()

print("="*70)
print("   HỆ THỐNG CELL-FREE MASSIVE MIMO")
print("="*70)
print(f"Số Access Points (M): {M}")
print(f"Số User Equipments (K): {K}")
print(f"Công suất tối đa mỗi AP (P_max): {P_MAX} mW")
print(f"Công suất nhiễu (σ²): {SIGMA2} mW")
print(f"Vùng phủ sóng: {AREA_SIZE}m x {AREA_SIZE}m")
print("="*70)

# ============================================
# 3. HÀM MỤC TIÊU: SUM-RATE
# ============================================
def calculate_sum_rate(power_vector):
    """
    Tính Sum-Rate của hệ thống theo công thức (2) trong báo cáo:
    R_sum = Σ_k log2(1 + SINR_k)
    
    GIẢI THÍCH CƠ BẢN:
    - Sum-Rate là tổng tốc độ truyền dữ liệu của tất cả người dùng
    - SINR = Signal-to-Interference-plus-Noise Ratio
    - Càng cao SINR, tốc độ truyền càng lớn
    
    CÔNG THỨC SINR_k:
    SINR_k = Tín_hiệu_mong_muốn / (Can_nhiễu_từ_UE_khác + Nhiễu_nền)
           = (Σ_m √p_mk * g_mk)² / (Σ_j≠k (Σ_m √p_mj * g_mj)² + σ²)
    
    Input: power_vector - vector công suất phẳng [M*K] (chromosome của GA)
    Output: Sum-Rate (bits/s/Hz) - fitness value
    """
    # Chuyển vector phẳng thành ma trận [M, K]
    # power_vector[0:M] = công suất AP 0 phát cho các UE
    # power_vector[M:2M] = công suất AP 1 phát cho các UE, v.v.
    P = power_vector.reshape(M, K)
    
    # Tính Sum-Rate = tổng rate của tất cả UE
    sum_rate = 0.0
    
    for k in range(K):  # Duyệt qua từng UE
        # BƯỚC 1: Tính tín hiệu mong muốn cho UE k
        # Tất cả APs đều phát tín hiệu cho UE k
        # Tín hiệu tổng = (Σ_m √p_mk * √β_mk)²
        signal_components = np.sqrt(P[:, k]) * np.sqrt(beta[:, k])
        desired_signal = np.sum(signal_components) ** 2
        
        # BƯỚC 2: Tính can nhiễu từ các UE khác
        # UE k nhận cả tín hiệu của mình và của các UE khác
        interference = 0.0
        for j in range(K):
            if j != k:  # Chỉ tính UE khác
                # Can nhiễu từ UE j đến UE k
                interferer_components = np.sqrt(P[:, j]) * np.sqrt(beta[:, k])
                interference += np.sum(interferer_components) ** 2
        
        # BƯỚC 3: Tính SINR và Rate
        sinr = desired_signal / (interference + SIGMA2)
        
        # Công thức Shannon: Rate = log2(1 + SINR)
        rate_k = np.log2(1 + sinr)
        
        # Cộng vào tổng Sum-Rate
        sum_rate += rate_k
    
    return sum_rate
    
    return sum_rate

def fitness_function(x):
    """
    Hàm fitness cho GA (chuyển từ bài toán maximize sang minimize)
    
    GIẢI THÍCH:
    - GA thường được thiết kế để tìm minimum
    - Bài toán ta muốn maximize Sum-Rate
    - Nên fitness = -Sum_Rate (maximize Sum-Rate = minimize -Sum-Rate)
    """
    return -calculate_sum_rate(x)

# ============================================
# 4. THUẬT TOÁN GENETIC ALGORITHM  
# ============================================
class GeneticAlgorithm:
    """
    Lớp Genetic Algorithm cho bài toán phân bổ công suất
    
    CÁC THÀNH PHẦN CHÍNH:
    1. Mã hóa: Real-coded (vector số thực)
    2. Fitness: Sum-Rate 
    3. Selection: Tournament Selection (k=3)
    4. Crossover: Arithmetic Crossover
    5. Mutation: Gaussian Mutation
    6. Constraint Handling: Repair Mechanism
    
    Theo cấu hình trong Bảng 1 của báo cáo
    """
    
    def __init__(self, pop_size=50, max_gen=100, pc=0.8, pm=0.2):
        """
        Khởi tạo tham số GA
        
        GIẢI THÍCH THAM SỐ:
        - pop_size: Kích thước quần thể (50 cá thể)
        - max_gen: Số thế hệ tối đa (100 thế hệ)  
        - pc: Xác suất lai ghép (80%)
        - pm: Xác suất đột biến (20%)
        - n_vars: Số biến = M×K = 50 (ma trận công suất phẳng)
        """
        self.pop_size = pop_size        
        self.n_vars = M * K             # Số biến quyết định (50 = 10×5)
        self.max_gen = max_gen          
        self.pc = pc                    
        self.pm = pm                    
        
        # Lưu lịch sử tiến hóa để vẽ biểu đồ
        self.best_fitness_history = []  # Fitness tốt nhất mỗi thế hệ
        self.avg_fitness_history = []   # Fitness trung bình mỗi thế hệ
        self.diversity_history = []     # Độ đa dạng quần thể
    
    def initialize_population(self):
        """
        BƯỚC 1: Khởi tạo quần thể ngẫu nhiên (Real-coded GA)
        
        GIẢI THÍCH:
        - Mỗi cá thể = 1 chromosome = 1 vector công suất [M×K]
        - Giá trị ngẫu nhiên trong [0, P_MAX/K] để tránh vi phạm ràng buộc ban đầu
        - P_MAX/K = chia đều công suất cho K người dùng
        
        Output: Ma trận quần thể [pop_size × n_vars]
        """
        print("🧬 Khởi tạo quần thể ngẫu nhiên...")
        
        # Tạo pop_size cá thể, mỗi cá thể có n_vars gen
        pop = np.random.uniform(0, P_MAX/K, (self.pop_size, self.n_vars))
        
        print(f"✅ Đã tạo {self.pop_size} cá thể, mỗi cá thể {self.n_vars} biến")
        return pop
    
    def tournament_selection(self, pop, fitness_values, k=3):
        """
        BƯỚC 2: Chọn lọc bằng Tournament Selection
        
        GIẢI THÍCH:
        - Chọn k=3 cá thể ngẫu nhiên từ quần thể  
        - Lấy cá thể có fitness tốt nhất trong 3 cá thể
        - Áp lực chọn lọc vừa phải (không quá mạnh như k=10, không quá yếu như k=2)
        
        Input: quần thể, fitness values, k=3
        Output: 1 cá thể cha/mẹ được chọn
        """
        # Chọn k chỉ số ngẫu nhiên
        indices = np.random.randint(0, len(pop), k)
        
        # Tìm cá thể tốt nhất (fitness nhỏ nhất vì ta minimize)
        best_idx = indices[np.argmin(fitness_values[indices])]
        
        # Trả về bản sao cá thể tốt nhất
        return pop[best_idx].copy()
    
    def arithmetic_crossover(self, parent1, parent2):
        """
        BƯỚC 3: Lai ghép bằng Arithmetic Crossover
        
        GIẢI THÍCH:
        - Kết hợp tuyến tính 2 cha mẹ với hệ số α ngẫu nhiên
        - child1 = α × parent1 + (1-α) × parent2  
        - child2 = α × parent2 + (1-α) × parent1
        - Phù hợp với Real-coded GA, đảm bảo con cái trong không gian khả thi
        
        Input: 2 cha mẹ
        Output: 2 con cái
        """
        # Hệ số lai ghép ngẫu nhiên α trong [0,1]
        alpha = np.random.rand()
        
        # Kiểm tra xác suất lai ghép pc
        if np.random.rand() < self.pc:
            # Thực hiện lai ghép
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        else:
            # Không lai ghép, trả về bản sao cha mẹ
            return parent1.copy(), parent2.copy()
    
    def gaussian_mutation(self, individual):
        """
        BƯỚC 4: Đột biến bằng Gaussian Mutation
        
        GIẢI THÍCH:
        - Với xác suất pm, thêm nhiễu Gaussian vào cá thể
        - Chỉ đột biến 10% gene để không phá hỏng quá nhiều
        - Nhiễu có độ lệch chuẩn = 10% giá trị tối đa
        - Giữ gene trong giới hạn [0, P_MAX/K]
        
        Input: 1 cá thể
        Output: cá thể sau đột biến
        """
        if np.random.rand() < self.pm:
            # Tạo nhiễu Gaussian có độ lệch chuẩn nhỏ
            noise = np.random.randn(len(individual)) * (P_MAX / K) * 0.1
            
            # Chọn ngẫu nhiên 10% gene để đột biến (không đột biến hết)
            mutation_mask = np.random.rand(len(individual)) < 0.1
            individual[mutation_mask] += noise[mutation_mask]
            
            # Giới hạn gene trong khoảng hợp lệ [0, P_MAX/K]
            individual = np.clip(individual, 0, P_MAX/K)
        
        return individual
    
    def repair_solution(self, individual):
        """
        BƯỚC 5: Sửa lỗi vi phạm ràng buộc (Repair Mechanism)
        
        GIẢI THÍCH RÀNG BUỘC:
        - Mỗi AP m có công suất tối đa P_MAX
        - Ràng buộc: Σ_k p_mk ≤ P_MAX (tổng công suất AP m cho tất cả UE ≤ P_MAX)
        - Nếu vi phạm → chia tỷ lệ (scaling) theo công thức (7) trong báo cáo
        
        CÔNG THỨC REPAIR:
        p_mk_new = p_mk × (P_MAX / Σ_k p_mk)  nếu Σ_k p_mk > P_MAX
        
        Input: cá thể có thể vi phạm ràng buộc
        Output: cá thể đã sửa, đảm bảo khả thi
        """
        # Chuyển vector phẳng thành ma trận [M, K]
        P = individual.reshape(M, K)
        
        # Kiểm tra và sửa ràng buộc cho từng AP
        for m in range(M):
            total_power_ap_m = np.sum(P[m, :])  # Tổng công suất AP m
            
            if total_power_ap_m > P_MAX:  # Vi phạm ràng buộc
                # Scaling: chia tỷ lệ để tổng = P_MAX
                scaling_factor = P_MAX / total_power_ap_m
                P[m, :] = P[m, :] * scaling_factor
        
        # Chuyển lại thành vector phẳng
        return P.flatten()
    
    def calculate_diversity(self, pop):
        """
        Tính độ đa dạng quần thể 
        
        GIẢI THÍCH:
        - Độ đa dạng = variance trung bình của các gene
        - Cao = quần thể đa dạng, thích hợp exploration
        - Thấp = quần thể hội tụ, thích hợp exploitation
        """
        return np.mean(np.var(pop, axis=0))
    
    def run(self, verbose=True):
        """
        HÀM CHÍNH: Chạy thuật toán GA
        
        QUY TRÌNH GA STANDARD:
        1. Khởi tạo quần thể ngẫu nhiên
        2. Lặp qua max_gen thế hệ:
           a. Repair ràng buộc
           b. Đánh giá fitness
           c. Selection (Tournament)
           d. Crossover (Arithmetic) 
           e. Mutation (Gaussian)
           f. Thay thế thế hệ (Generational)
        3. Trả về nghiệm tốt nhất
        """
        
        if verbose:
            print("\n" + "="*70)
            print("   BẮT ĐẦU GENETIC ALGORITHM")
            print("="*70)
            print(f"Kích thước quần thể: {self.pop_size}")
            print(f"Số thế hệ: {self.max_gen}")
            print(f"Xác suất lai ghép: {self.pc}")
            print(f"Xác suất đột biến: {self.pm}")
            print(f"Số biến tối ưu: {self.n_vars} (ma trận {M}×{K})")
            print("-"*70)
        
        # BƯỚC 1: Khởi tạo quần thể
        pop = self.initialize_population()
        
        # Theo dõi nghiệm tốt nhất
        best_solution = None
        best_fitness = float('inf')  # +∞ vì ta minimize
        
        # BƯỚC 2: Vòng lặp chính - Tiến hóa qua các thế hệ
        for gen in range(self.max_gen):
            # BƯỚC 2a: Sửa lỗi vi phạm ràng buộc cho toàn quần thể
            pop = np.array([self.repair_solution(ind) for ind in pop])
            
            # BƯỚC 2b: Đánh giá fitness cho toàn quần thể
            fitness_values = np.array([fitness_function(ind) for ind in pop])
            
            # BƯỚC 2c: Cập nhật nghiệm tốt nhất
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_solution = pop[current_best_idx].copy()
            
            # BƯỚC 2d: Lưu thống kê để vẽ biểu đồ
            self.best_fitness_history.append(-best_fitness)  # Chuyển về Sum-Rate
            self.avg_fitness_history.append(-np.mean(fitness_values))
            self.diversity_history.append(self.calculate_diversity(pop))
            
            # In tiến trình mỗi 20 thế hệ
            if verbose and (gen % 20 == 0 or gen == self.max_gen - 1):
                print(f"Thế hệ {gen:3d}: Sum-Rate tốt nhất = {-best_fitness:.4f} bits/s/Hz")
            
            # BƯỚC 2e: Tạo thế hệ mới (Generational Replacement)
            new_pop = []
            
            # Elitism: Giữ lại 10% cá thể tốt nhất để không mất nghiệm tốt
            elite_count = int(0.1 * self.pop_size) 
            elite_indices = np.argsort(fitness_values)[:elite_count]  # Chỉ số cá thể tốt nhất
            for idx in elite_indices:
                new_pop.append(pop[idx].copy())
            
            # Sinh sản để tạo đủ pop_size cá thể mới
            while len(new_pop) < self.pop_size:
                # Selection: Chọn 2 cha mẹ bằng Tournament Selection
                parent1 = self.tournament_selection(pop, fitness_values, k=3)
                parent2 = self.tournament_selection(pop, fitness_values, k=3)
                
                # Crossover: Lai ghép tạo 2 con
                child1, child2 = self.arithmetic_crossover(parent1, parent2)
                
                # Mutation: Đột biến 2 con
                child1 = self.gaussian_mutation(child1)
                child2 = self.gaussian_mutation(child2)
                
                # Thêm con vào quần thể mới
                new_pop.extend([child1, child2])
            
            # Cắt về đúng kích thước quần thể (phòng trường hợp thừa)
            pop = np.array(new_pop[:self.pop_size])
        
        # BƯỚC 3: Kết thúc thuật toán
        if verbose:
            print("="*70)
            print("   KẾT THÚC THUẬT TOÁN")
            print(f"✅ Nghiệm tốt nhất: Sum-Rate = {-best_fitness:.4f} bits/s/Hz")
            print("="*70)
        
        # Trả về: nghiệm tốt nhất, fitness, thống kê
        return best_solution, -best_fitness, {
            'best_history': self.best_fitness_history,
            'avg_history': self.avg_fitness_history,
            'diversity_history': self.diversity_history
        }

# ============================================
# 5. BASELINE: PHƯƠNG PHÁP THAM CHIẾU
# ============================================
def equal_power_allocation():
    """
    Phương pháp tham chiếu: Phân bổ công suất đều
    
    GIẢI THÍCH:
    - Phương pháp đơn giản nhất: chia đều công suất
    - Mỗi AP phân bổ P_MAX/K cho từng UE
    - p_mk = P_MAX/K, ∀m, k
    - Dùng để so sánh hiệu quả của GA
    
    Output: vector công suất đều, Sum-Rate tương ứng
    """
    print("📊 Tính toán phương pháp phân bổ đều (Baseline)...")
    
    # Tạo ma trận công suất đều
    P_equal = np.ones((M, K)) * (P_MAX / K)
    
    # Tính Sum-Rate của phương pháp này
    sum_rate_equal = calculate_sum_rate(P_equal.flatten())
    
    print(f"✅ Sum-Rate phân bổ đều: {sum_rate_equal:.4f} bits/s/Hz")
    
    return P_equal.flatten(), sum_rate_equal

# ============================================
# 6. HÀM VẼ ĐỒ THỊ (THEO HÌNH TRONG BÁO CÁO)  
# ============================================
def plot_convergence(stats, baseline_rate):
    """
    Vẽ Hình 1: Biểu đồ hội tụ của hàm mục tiêu Sum-Rate
    
    GIẢI THÍCH BIỂU ĐỒ:
    - Trục x: Số thế hệ (0 → max_gen)
    - Trục y: Sum-Rate (bits/s/Hz) 
    - Đường đỏ: Sum-Rate tốt nhất mỗi thế hệ
    - Đường xanh: Sum-Rate trung bình quần thể
    - Đường ngang: Baseline (phân bổ đều)
    
    Hình 1: Biểu đồ hội tụ (convergence.png)
    Theo mô tả trong Mục 3.4.1 của báo cáo
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = range(len(stats['best_history']))
    
    # Đường GA - Best
    ax.plot(generations, stats['best_history'], 'b-', linewidth=2.5, 
            label='GA - Sum-Rate tốt nhất', marker='o', markersize=4, markevery=10)
    
    # Đường GA - Average
    ax.plot(generations, stats['avg_history'], 'g--', linewidth=2, 
            label='GA - Sum-Rate trung bình', alpha=0.7)
    
    # Đường Baseline
    ax.axhline(y=baseline_rate, color='r', linestyle=':', linewidth=2.5, 
               label='Phân bổ đều (Baseline)')
    
    ax.set_xlabel('Thế hệ (Generation)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sum-Rate (bits/s/Hz)', fontsize=14, fontweight='bold')
    ax.set_title('Biểu đồ hội tụ của hàm mục tiêu Sum-Rate theo số thế hệ', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=300, bbox_inches='tight')
    print("✓ Đã lưu: convergence.png")
    plt.close()

def plot_heatmap(power_ga, power_equal):
    """
    Hình 2: Bản đồ phân bổ công suất (heatmap.png)
    Theo mô tả trong Mục 3.4.2 của báo cáo
    """
    power_ga = power_ga.reshape(M, K)
    power_equal = power_equal.reshape(M, K)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap GA
    im1 = ax1.imshow(power_ga, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('User Equipment (UE)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Access Point (AP)', fontsize=13, fontweight='bold')
    ax1.set_title('GA - Phân bổ công suất tối ưu', fontsize=15, fontweight='bold')
    ax1.set_xticks(range(K))
    ax1.set_xticklabels([f'UE{k+1}' for k in range(K)])
    ax1.set_yticks(range(M))
    ax1.set_yticklabels([f'AP{m+1}' for m in range(M)])
    
    # Thêm giá trị vào ô
    for m in range(M):
        for k in range(K):
            text = ax1.text(k, m, f'{power_ga[m, k]:.1f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Công suất (mW)', fontsize=12, fontweight='bold')
    
    # Heatmap Baseline
    im2 = ax2.imshow(power_equal, cmap='Blues', aspect='auto', interpolation='nearest')
    ax2.set_xlabel('User Equipment (UE)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Access Point (AP)', fontsize=13, fontweight='bold')
    ax2.set_title('Phân bổ đều (Baseline)', fontsize=15, fontweight='bold')
    ax2.set_xticks(range(K))
    ax2.set_xticklabels([f'UE{k+1}' for k in range(K)])
    ax2.set_yticks(range(M))
    ax2.set_yticklabels([f'AP{m+1}' for m in range(M)])
    
    # Thêm giá trị vào ô
    for m in range(M):
        for k in range(K):
            text = ax2.text(k, m, f'{power_equal[m, k]:.1f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Công suất (mW)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Đã lưu: heatmap.png")
    plt.close()

def plot_comparison(ga_rate, baseline_rate):
    """
    Hình 3: So sánh hiệu năng (comparison.png)
    Theo mô tả trong Mục 3.4.3 của báo cáo
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Phân bổ đều\n(Baseline)', 'Genetic Algorithm\n(Đề xuất)']
    rates = [baseline_rate, ga_rate]
    colors = ['#ff6b6b', '#4ecdc4']
    
    bars = ax.bar(methods, rates, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=2.5, width=0.5)
    
    # Thêm giá trị trên cột
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.4f}\nbits/s/Hz',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Tính % cải thiện
    improvement = (rates[1] - rates[0]) / rates[0] * 100
    ax.text(0.5, max(rates) * 0.6, 
            f'Cải thiện: +{improvement:.2f}%',
            ha='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='yellow', alpha=0.8, 
                     edgecolor='black', linewidth=2))
    
    ax.set_ylabel('Sum-Rate (bits/s/Hz)', fontsize=14, fontweight='bold')
    ax.set_title('So sánh hiệu năng Sum-Rate cuối cùng', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, max(rates) * 1.2)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Đã lưu: comparison.png")
    plt.close()

# ============================================
# 7. CHƯƠNG TRÌNH CHÍNH
# ============================================
if __name__ == "__main__":
    print("\n" + "🎯 "*30)
    print("   GENETIC ALGORITHM CHO BÀI TOÁN PHÂN BỔ CÔNG SUẤT")
    print("   TRONG MẠNG CELL-FREE MASSIVE MIMO")
    print("🎯 "*30 + "\n")
    
    # Bước 1: Tính Baseline
    print("📊 BƯỚC 1: Tính phương pháp phân bổ đều (Baseline)...")
    print("-"*70)
    power_baseline, rate_baseline = equal_power_allocation()
    print(f"Sum-Rate (Phân bổ đều): {rate_baseline:.4f} bits/s/Hz")
    
    # Bước 2: Chạy GA
    print("\n📊 BƯỚC 2: Chạy Genetic Algorithm...")
    print("-"*70)
    ga = GeneticAlgorithm(
        pop_size=50,
        max_gen=100,
        pc=0.8,
        pm=0.2
    )
    
    best_power, best_rate, stats = ga.run(verbose=True)
    
    # Bước 3: So sánh kết quả
    print("\n" + "="*70)
    print("   KẾT QUẢ CUỐI CÙNG")
    print("="*70)
    print(f"Genetic Algorithm:      {best_rate:.4f} bits/s/Hz")
    print(f"Phân bổ đều (Baseline): {rate_baseline:.4f} bits/s/Hz")
    improvement = (best_rate - rate_baseline) / rate_baseline * 100
    print(f"Mức cải thiện:          +{improvement:.2f}%")
    print("="*70)
    
    # Bước 4: Vẽ đồ thị
    print("\n📊 BƯỚC 3: Tạo các biểu đồ cho báo cáo...")
    print("-"*70)
    plot_convergence(stats, rate_baseline)
    plot_heatmap(best_power, power_baseline)
    plot_comparison(best_rate, rate_baseline)
    
    print("\n" + "="*70)
    print("   ✅ HOÀN THÀNH!")
    print("="*70)
    print("\n📁 Các file đã tạo:")
    print("   1. convergence.png  - Biểu đồ hội tụ (Hình 1)")
    print("   2. heatmap.png      - Bản đồ phân bổ công suất (Hình 2)")
    print("   3. comparison.png   - So sánh hiệu năng (Hình 3)")
    
    print("\n💡 CHÚ Ý KHI TRÌNH BÀY:")
    print("   • Giải thích rõ ràng mô hình hệ thống Cell-Free")
    print("   • Trình bày công thức SINR và Sum-Rate")
    print("   • Nhấn mạnh cơ chế Repair để đảm bảo ràng buộc")
    print("   • Phân tích ý nghĩa của Heatmap (User-centric)")
    print("   • So sánh với Baseline để thấy hiệu quả của GA")
    
    print("\n📚 Tài liệu tham khảo đề xuất:")
    print("   [1] Ngo et al. (2017), 'Cell-Free Massive MIMO'")
    print("   [2] Goldberg (1989), 'Genetic Algorithms'")
    print("   [3] Whitley (1994), 'A Genetic Algorithm Tutorial'")
    
    print("\n🎓 Chúc em thuyết trình tốt!\n")
