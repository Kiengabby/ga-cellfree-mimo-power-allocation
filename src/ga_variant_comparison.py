# ============================================================
# SO SÁNH STANDARD GA vs ADAPTIVE GA (BIẾN THỂ CẢII THIỆN)
# ============================================================
# MỤC ĐÍCH: 
# - So sánh GA gốc với biến thể cải tiến
# - Chứng minh biến thể tốt hơn
# - Tạo biểu đồ cho báo cáo
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Cấu hình font
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

# ============================================
# CẤU HÌNH HỆ THỐNG
# ============================================
M, K = 10, 5        # 10 APs, 5 UEs
P_MAX = 200.0       # Công suất tối đa (mW)
np.random.seed(42)  # Để kết quả giống nhau

# Tạo hệ số kênh truyền ngẫu nhiên
beta = np.random.uniform(0.1, 1.0, (M, K))
for k in range(K): 
    beta[:, k] = beta[:, k] ** 3  # Tăng độ chênh lệch

# ============================================
# HÀM TÍNH SUM-RATE
# ============================================
def calculate_sum_rate(p_vec):
    """Tính tổng tốc độ truyền của hệ thống"""
    P = np.abs(p_vec.reshape(M, K))  # Đảm bảo không âm
    P = np.clip(P, 0, P_MAX)         # Giới hạn trong [0, P_MAX]
    
    rate = 0
    for k in range(K):
        # Tín hiệu mong muốn
        sig = np.sum(np.sqrt(P[:, k]) * np.sqrt(beta[:, k]))**2
        # Can nhiễu
        inter = 0
        for j in range(K):
            if j != k: 
                inter += np.sum(np.sqrt(P[:, j]) * np.sqrt(beta[:, k]))**2
        # Rate
        rate += np.log2(1 + sig/(inter + 1.0))
    return rate

# ============================================
# STANDARD GA (THUẬT TOÁN GỐC)
# ============================================
class StandardGA:
    """
    GA chuẩn với:
    - Tỷ lệ đột biến cố định: pm = 0.1
    - Không có elitism 
    - Tournament-2 selection
    """
    def __init__(self, pop_size=50, max_gen=100, pc=0.8, pm=0.1):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pc = pc        # Xác suất lai ghép
        self.pm = pm        # Xác suất đột biến (CỐ ĐỊNH)
        self.history = []   # Lưu lịch sử để vẽ biểu đồ

    def run(self):
        """Chạy Standard GA"""
        # Khởi tạo quần thể ngẫu nhiên
        pop = np.random.uniform(0, P_MAX/K, (self.pop_size, M*K))
        
        for gen in range(self.max_gen):
            # Sửa vi phạm ràng buộc
            for i in range(self.pop_size):
                P = pop[i].reshape(M, K)
                for m in range(M):
                    if np.sum(P[m,:]) > P_MAX: 
                        P[m,:] *= (P_MAX / np.sum(P[m,:]))
                pop[i] = P.flatten()
            
            # Đánh giá fitness
            scores = np.array([calculate_sum_rate(ind) for ind in pop])
            self.history.append(np.max(scores))  # Lưu best của thế hệ
            
            # Tạo thế hệ mới (Standard GA)
            new_pop = []
            while len(new_pop) < self.pop_size:
                # Tournament selection (k=2)
                ids = np.random.randint(0, self.pop_size, 2)
                p1 = pop[ids[np.argmax(scores[ids])]]
                ids = np.random.randint(0, self.pop_size, 2)
                p2 = pop[ids[np.argmax(scores[ids])]]
                
                # Lai ghép với xác suất pc
                if np.random.rand() < self.pc:
                    alpha = np.random.rand()
                    c1 = alpha*p1 + (1-alpha)*p2
                else: 
                    c1 = p1.copy()
                
                # Đột biến với xác suất pm CỐ ĐỊNH
                if np.random.rand() < self.pm:
                    c1 += np.random.randn(M*K) * 5
                    c1 = np.clip(c1, 0, P_MAX)
                
                new_pop.append(c1)
            pop = np.array(new_pop)
        
        return self.history

# ============================================
# ADAPTIVE GA (BIẾN THỂ CẢI THIỆN)
# ============================================
class AdaptiveGA:
    """
    GA cải tiến với:
    - Tỷ lệ đột biến thích ứng: pm từ 0.5 → 0.01 
    - Elitism: giữ lại cá thể tốt nhất
    - Tournament-3 selection (áp lực cao hơn)
    - Fine-tuning: nhiễu giảm dần theo thời gian
    """
    def __init__(self, pop_size=50, max_gen=100):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.history = []
        
    def run(self):
        """Chạy Adaptive GA"""
        # Khởi tạo quần thể
        pop = np.random.uniform(0, P_MAX/K, (self.pop_size, M*K))
        
        # Tham số thích ứng
        pm_start, pm_end = 0.5, 0.01  # Đột biến giảm từ 50% → 1%
        
        for gen in range(self.max_gen):
            # Tính pm thích ứng theo thời gian
            current_pm = pm_start - (gen/self.max_gen) * (pm_start - pm_end)
            
            # Sửa vi phạm ràng buộc
            for i in range(self.pop_size):
                P = pop[i].reshape(M, K)
                for m in range(M):
                    if np.sum(P[m,:]) > P_MAX: 
                        P[m,:] *= (P_MAX / np.sum(P[m,:]))
                pop[i] = P.flatten()
                
            # Đánh giá fitness
            scores = np.array([calculate_sum_rate(ind) for ind in pop])
            best_val = np.max(scores)
            self.history.append(best_val)
            
            # ELITISM: Giữ lại cá thể tốt nhất (không qua lai ghép)
            new_pop = [pop[np.argmax(scores)]] 
            
            # Tạo phần còn lại của thế hệ mới
            while len(new_pop) < self.pop_size:
                # Tournament-3 selection (áp lực cao hơn)
                ids = np.random.randint(0, self.pop_size, 3)
                p1 = pop[ids[np.argmax(scores[ids])]]
                ids = np.random.randint(0, self.pop_size, 3)
                p2 = pop[ids[np.argmax(scores[ids])]]
                
                # Lai ghép
                alpha = np.random.rand()
                c1 = alpha*p1 + (1-alpha)*p2
                
                # Đột biến thích ứng với nhiễu giảm dần
                if np.random.rand() < current_pm:
                    # Nhiễu giảm theo thời gian (fine-tuning)
                    noise_scale = 5.0 * (1 - gen/self.max_gen) 
                    c1 += np.random.randn(M*K) * noise_scale
                    c1 = np.clip(c1, 0, P_MAX)  # Clip giá trị hợp lệ
                
                new_pop.append(c1)
            pop = np.array(new_pop)
        
        return self.history

# 3. CLASS ADAPTIVE GA (BIẾN THỂ - MỤC 4)
class AdaptiveGA:
    def __init__(self, pop_size=50, max_gen=100):
        self.pop_size, self.max_gen = pop_size, max_gen
        self.history = []
        
    def run(self):
        pop = np.random.uniform(0, P_MAX/K, (self.pop_size, M*K))
        
        # Adaptive Mutation Rate: Giảm dần theo thời gian (Explore -> Exploit)
        pm_start, pm_end = 0.5, 0.01 
        
        for gen in range(self.max_gen):
            # Tính Pm thích nghi
            current_pm = pm_start - (gen/self.max_gen) * (pm_start - pm_end)
            
            # Repair
            for i in range(self.pop_size):
                P = pop[i].reshape(M, K)
                for m in range(M):
                    if np.sum(P[m,:]) > P_MAX: P[m,:] *= (P_MAX / np.sum(P[m,:]))
                pop[i] = P.flatten()
                
            scores = np.array([calculate_sum_rate(ind) for ind in pop])
            best_val = np.max(scores)
            self.history.append(best_val)
            
            # ELITISM: Giữ lại con tốt nhất ngay lập tức (Không qua lai ghép)
            new_pop = [pop[np.argmax(scores)]] 
            
            while len(new_pop) < self.pop_size:
                # Tournament
                ids = np.random.randint(0, self.pop_size, 3)
                parent_idx = ids[np.argmax(scores[ids])]
                p1 = pop[parent_idx]
                
                ids = np.random.randint(0, self.pop_size, 3)
                p2 = pop[ids[np.argmax(scores[ids])]]
                
                # Arithmetic Crossover
                alpha = np.random.rand()
                c1 = alpha*p1 + (1-alpha)*p2
                
                # Adaptive Mutation
                if np.random.rand() < current_pm:
                    # Càng về sau nhiễu càng nhỏ (Fine-tuning)
                    noise_scale = 5.0 * (1 - gen/self.max_gen) 
                    c1 += np.random.randn(M*K) * noise_scale
                    c1 = np.clip(c1, 0, P_MAX)  # Clip giá trị hợp lệ
                
                new_pop.append(c1)
            pop = np.array(new_pop)
        return self.history

# 4. CHẠY SO SÁNH
print("="*60)
print("   SO SÁNH STANDARD GA vs ADAPTIVE GA")
print("="*60)
print("Đang chạy Standard GA...")
std_ga = StandardGA()
hist_std = std_ga.run()

print("Đang chạy Adaptive GA (Biến thể)...")
ada_ga = AdaptiveGA()
hist_ada = ada_ga.run()

# 5. VẼ ĐỒ THỊ SO SÁNH
plt.figure(figsize=(12, 7))
plt.plot(hist_std, 'r--', linewidth=2.5, label='Standard GA (Mục 3)', alpha=0.8)
plt.plot(hist_ada, 'b-', linewidth=3, label='Adaptive GA + Elitism (Mục 4)', alpha=0.9)

plt.title('So sánh hiệu năng: Thuật toán gốc vs. Biến thể cải tiến', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Thế hệ (Generations)', fontsize=14, fontweight='bold')
plt.ylabel('Sum-Rate (bits/s/Hz)', fontsize=14, fontweight='bold')

# Thêm grid và legend đẹp hơn
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=13, loc='lower right', framealpha=0.9)

# Thêm annotation hiệu năng cuối
final_std = hist_std[-1]
final_ada = hist_ada[-1]
improvement = (final_ada - final_std) / final_std * 100

plt.annotate(f'Standard GA\nKết quả cuối: {final_std:.3f}', 
             xy=(len(hist_std)-1, final_std), xytext=(70, final_std-0.5),
             arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
             fontsize=11, ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.annotate(f'Adaptive GA\nKết quả cuối: {final_ada:.3f}\nCải thiện: +{improvement:.1f}%', 
             xy=(len(hist_ada)-1, final_ada), xytext=(70, final_ada+0.3),
             arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
             fontsize=11, ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

plt.tight_layout()
plt.savefig('variant_comparison.png', dpi=300, bbox_inches='tight')

# 6. KẾT QUẢ
print("\n" + "="*60)
print("   KẾT QUẢ SO SÁNH")
print("="*60)
print(f"Standard GA (cuối):      {final_std:.4f} bits/s/Hz")
print(f"Adaptive GA (cuối):      {final_ada:.4f} bits/s/Hz")
print(f"Cải thiện:              +{improvement:.2f}%")
print("="*60)
print("✓ Đã lưu: variant_comparison.png")
print("\n💡 GIẢI THÍCH BIẾN THỂ:")
print("  • Adaptive Mutation: pm giảm từ 0.5 → 0.01 (Explore → Exploit)")
print("  • Elitism: Giữ nguyên cá thể tốt nhất mỗi thế hệ")
print("  • Tournament k=3: Tăng áp lực chọn lọc")
print("  • Fine-tuning: Nhiễu đột biến giảm dần theo thời gian")
print("🎓 Sử dụng ảnh này để trình bày Mục 4 trong báo cáo!\n")
