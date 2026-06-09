import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 读取四列对数 RAR 数据
# 数据来源：SPARC 径向加速度关系公开数据
# 文件格式：gbar_log, e_gbar, gobs_log, e_gobs（均为 log10 值）
# ============================================================
file_path = r"data/Sparc_Radial_Acceleration_Relation.mrt"

data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # 跳过空行、注释(#)、表格符号(\ |)
        # 注意：数据值可能为负数（log10 值），不能跳过负号开头的行
        if not line or line.startswith('#') or line.startswith('\\') or line.startswith('|'):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                gbar_log = float(parts[0])
                e_gbar   = float(parts[1])
                gobs_log = float(parts[2])
                e_gobs   = float(parts[3])
                data.append([gbar_log, e_gbar, gobs_log, e_gobs])
            except ValueError:
                continue

data = np.array(data)
if len(data) == 0:
    print("未读取到有效数据。请检查文件是否放在桌面，并命名为 Sparc_Radial_Acceleration_Relation.mrt")
    exit()

gbar_log = data[:, 0]
gobs_log = data[:, 2]
gbar = 10 ** gbar_log    # 转为线性值 m/s^2
gobs = 10 ** gobs_log

print(f"成功读取 {len(data)} 个数据点")

# ============================================================
# 2. 相变标量场论 ν 函数（最终定稿版）
# 参数 α=3.01, β=1.54, yc=0.48 由 SPARC 175 星系全局拟合确定
# 强场极限 y≫yc：ν→1，恢复牛顿/GR
# 弱场极限 y≪yc：ν→1+α，引力增强有限饱和
# ============================================================
A0 = 1.2e-10   # m/s²，临界加速度

def nu_phase(y, alpha=3.01, beta=1.54, yc=0.48):
    """相变标量场论 ν 函数"""
    return 1.0 + alpha / (1.0 + (y / yc)**beta)

def predicted_gobs(gbar, a0=A0):
    """理论预测的总加速度：g_obs = g_bar × ν(g_bar/a0)"""
    y = gbar / a0
    return gbar * nu_phase(y)

# ============================================================
# 3. 绘图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：RAR 散点密度图 + 理论曲线
ax = axes[0]
hb = ax.hexbin(gbar, gobs, gridsize=50, bins='log', cmap='viridis', mincnt=1)
plt.colorbar(hb, ax=ax, label='Number of points')

gbar_range = np.logspace(-13, -8, 200)
gobs_pred = predicted_gobs(gbar_range)
ax.plot(gbar_range, gobs_pred, 'r-', linewidth=2, label=f'Phase-transition ν (α=3.01, β=1.54, yc=0.48)')
ax.plot(gbar_range, gbar_range, 'k--', alpha=0.5, label='Newtonian (no DM)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Baryonic acceleration g_bar (m/s²)', fontsize=12)
ax.set_ylabel('Observed acceleration g_obs (m/s²)', fontsize=12)
ax.set_title('SPARC Radial Acceleration Relation', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# 右图：残差（g_obs / g_theory）
ax2 = axes[1]
gobs_pred_at_data = predicted_gobs(gbar)
ratio = gobs / gobs_pred_at_data
hb2 = ax2.hexbin(gbar, ratio, gridsize=40, bins='log', cmap='coolwarm', mincnt=1)
plt.colorbar(hb2, ax=ax2, label='Points')
ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect match')
ax2.set_xscale('log')
ax2.set_xlabel('g_bar (m/s²)', fontsize=12)
ax2.set_ylabel('g_obs / g_theory', fontsize=12)
ax2.set_title('Residuals of the RAR', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')
ax2.set_ylim(0.5, 2.0)

plt.tight_layout()
plt.savefig('RAR_phase_transition.png', dpi=150)
plt.show()

# ============================================================
# 4. 统计输出
# 中位数反映理论预测的整体偏差（接近1表示无系统性偏移）
# 对数散射反映数据点围绕理论曲线的弥散程度
# SPARC 官方报告的内在散射约 0.12-0.13 dex
# ============================================================
log_ratio = np.log10(ratio)
scatter = np.std(log_ratio)
median_ratio = np.median(ratio)
print(f"\n--- RAR 验证结果 ---")
print(f"数据点总数: {len(data)}")
print(f"g_obs / g_theory 中位数: {median_ratio:.3f}")
print(f"对数散射 (dex): {scatter:.3f}")
print(f"参考：SPARC 官方报告内在散射约 0.12-0.13 dex")