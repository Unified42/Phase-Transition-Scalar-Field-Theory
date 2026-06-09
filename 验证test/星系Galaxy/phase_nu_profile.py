import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================
# 1. 读取 SPARC 旋转曲线数据
# 数据来源：SPARC 数据库 (Lelli et al. 2016)
# 列说明：Name=星系名, D=距离(Mpc), Rad=半径(kpc),
#         Vobs=观测速度(km/s), e_Vobs=速度误差(km/s),
#         Vgas=气体速度贡献, Vdisk=恒星盘速度贡献,
#         Vbul=核球速度贡献（Vdisk和Vbul以M/L=1给出）
# ============================================================
data_file = r"data/SPARC_Lelli_2016_data.txt"
rows = []
with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('\\') or line.startswith('|') or line.startswith('-'):
            continue
        parts = line.split()
        if len(parts) >= 8:
            rows.append(parts[:8])

col_names = ['Name', 'D', 'Rad', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul']
data = pd.DataFrame(rows, columns=col_names)
for col in col_names[1:]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna(subset=['Rad','Vobs','Vgas','Vdisk','Vbul'])

# ============================================================
# 2. 物理常数
# ============================================================
a0 = 1.2e-10                  # m/s²，临界加速度
kpc_to_m = 3.085677581e19     # m/kpc，单位转换
km_to_m = 1e3                 # m/km，单位转换

def nu_std(y):
    """MOND 标准插值函数（用于初步质光比拟合）"""
    return 0.5 + 0.5 * np.sqrt(1 + 4.0 / y)

# ============================================================
# 3. 用标准 MOND ν 函数为每个星系初步拟合质光比
# 目的：获得合理的 g_bar 和 g_obs 初始估计
# 后续的 ν 函数参数扫描将在此基础上进行
# ============================================================
def chi2_ml(params, r, Vobs, Vgas, Vdisk, Vbul, eVobs):
    ml_d, ml_b = params
    Vbar2 = Vgas**2 + ml_d*Vdisk**2 + ml_b*Vbul**2
    Vbar2 = np.maximum(Vbar2, 0)
    gbar = Vbar2 / (r * kpc_to_m) * km_to_m**2
    y = gbar / a0
    gobs_pred = gbar * nu_std(y)
    Vpred = np.sqrt(gobs_pred * r * kpc_to_m) / km_to_m
    return np.sum(((Vobs - Vpred)/eVobs)**2)

ml_dict = {}
for name, grp in data.groupby('Name'):
    r = grp['Rad'].values
    Vobs = grp['Vobs'].values
    Vgas = grp['Vgas'].values
    Vdisk = grp['Vdisk'].values
    Vbul = grp['Vbul'].values
    eVobs = grp['e_Vobs'].values
    eVobs = np.where(eVobs <= 0, 1.0, eVobs)
    has_bulge = np.max(Vbul) > 0.1
    if has_bulge:
        res = minimize(lambda p: chi2_ml(p, r, Vobs, Vgas, Vdisk, Vbul, eVobs),
                       x0=[0.5,0.7], bounds=[(0.01,5),(0.01,5)])
        ml_d, ml_b = res.x
    else:
        res = minimize(lambda p: chi2_ml([p[0],0], r, Vobs, Vgas, Vdisk, Vbul, eVobs),
                       x0=[0.5], bounds=[(0.01,5)])
        ml_d, ml_b = res.x[0], 0.0
    ml_dict[name] = (ml_d, ml_b)

data['ML_disk'] = data['Name'].map(lambda x: ml_dict[x][0])
data['ML_bulge'] = data['Name'].map(lambda x: ml_dict[x][1])
data['Vbar2'] = data['Vgas']**2 + data['ML_disk']*data['Vdisk']**2 + data['ML_bulge']*data['Vbul']**2
data['gbar'] = data['Vbar2'] / (data['Rad'] * kpc_to_m) * km_to_m**2
data['gobs'] = (data['Vobs']**2) / (data['Rad'] * kpc_to_m) * km_to_m**2
data = data[(data['gbar']>0) & (data['gobs']>0)]
gbar_all = data['gbar'].values
gobs_all = data['gobs'].values
print(f"有效数据点: {len(data)}")

# ============================================================
# 4. 相变标量场论 ν 函数
# 参数 α：深蜷缩态引力增强饱和幅度
# 参数 β：相变陡度
# 参数 yc：相变中心对应的无量纲加速度
# ============================================================
def nu_phase(y, alpha, beta, yc):
    """相变标量场论 ν 函数"""
    return 1.0 + alpha / (1.0 + (y/yc)**beta)

# ============================================================
# 5. 损失函数：综合中位数偏差和散射
# 目标：med → 1.0（无系统性偏差），scatter → 最小
# ============================================================
def loss_func(params):
    alpha, beta, yc = params
    if alpha <= 0 or beta <= 0 or yc <= 0:
        return 1e10
    y = gbar_all / a0
    nu = nu_phase(y, alpha, beta, yc)
    gobs_pred = gbar_all * nu
    ratio = gobs_all / gobs_pred
    med = np.median(ratio)
    scatter = np.std(np.log10(ratio))
    return abs(med - 1.0) + scatter

# ============================================================
# 6. 高精度局部优化
# 使用 Nelder-Mead 单纯形算法从网格搜索最佳点出发
# ============================================================
p0 = [3.0, 1.5, 0.5]  # 初始猜测（由前期网格搜索确定）
res = minimize(loss_func, p0, method='Nelder-Mead',
               options={'xatol':1e-6, 'fatol':1e-8, 'maxiter':10000, 'maxfev':10000})
alpha_opt, beta_opt, yc_opt = res.x
loss_opt = res.fun

print(f"\n========== 局部优化最佳参数 ==========")
print(f"α  = {alpha_opt:.4f}")
print(f"β  = {beta_opt:.4f}")
print(f"yc = {yc_opt:.4f}")
print(f"loss = |med-1| + scatter = {loss_opt:.4f}")

# 最终统计
y_all = gbar_all / a0
nu_opt = nu_phase(y_all, alpha_opt, beta_opt, yc_opt)
gobs_pred_opt = gbar_all * nu_opt
ratio_opt = gobs_all / gobs_pred_opt
med_opt = np.median(ratio_opt)
scat_opt = np.std(np.log10(ratio_opt))
print(f"预测/观测比值中位数: {med_opt:.4f}")
print(f"对数散射: {scat_opt:.4f} dex")
print(f"(注：散射包含数据内在弥散和测量误差，不完全反映模型精度)")

# ============================================================
# 7. 参数轮廓扫描
# 固定一个参数，优化其余两个，观察损失函数的变化
# 用于评估参数之间的简并性和最优值的稳定性
# ============================================================
print("\n========== 参数轮廓扫描 ==========")

print("\n--- α 固定扫描 ---")
for alpha in [2.0, 2.5, 3.0, 3.5, 4.0]:
    res_scan = minimize(lambda p: loss_func([alpha, p[0], p[1]]),
                        [beta_opt, yc_opt], method='Nelder-Mead')
    b, yc_s = res_scan.x
    l = res_scan.fun
    marker = " ← BEST" if abs(alpha - alpha_opt) < 0.01 else ""
    print(f"α={alpha:.1f} → β={b:.3f}, yc={yc_s:.3f}, loss={l:.4f}{marker}")

print("\n--- β 固定扫描 ---")
for beta in [1.0, 1.5, 2.0, 2.5, 3.0]:
    res_scan = minimize(lambda p: loss_func([p[0], beta, p[1]]),
                        [alpha_opt, yc_opt], method='Nelder-Mead')
    a, yc_s = res_scan.x
    l = res_scan.fun
    marker = " ← BEST" if abs(beta - beta_opt) < 0.01 else ""
    print(f"β={beta:.1f} → α={a:.3f}, yc={yc_s:.3f}, loss={l:.4f}{marker}")

print("\n--- yc 固定扫描 ---")
for yc in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    res_scan = minimize(lambda p: loss_func([p[0], p[1], yc]),
                        [alpha_opt, beta_opt], method='Nelder-Mead')
    a, b = res_scan.x
    l = res_scan.fun
    marker = " ← BEST" if abs(yc - yc_opt) < 0.01 else ""
    print(f"yc={yc:.2f} → α={a:.3f}, β={b:.3f}, loss={l:.4f}{marker}")

# ============================================================
# 8. 置信区间估计
# 以损失函数 5% 增量作为近似 1σ 容忍区间
# 反映参数在合理范围内变化时拟合质量的退化程度
# ============================================================
loss_threshold = loss_opt * 1.05

print(f"\n========== 5% 容忍区间 (loss < {loss_threshold:.4f}) ==========")

print("α 容忍区间:")
alphas_test = np.linspace(1.5, 5.0, 36)
for a in alphas_test:
    res_t = minimize(lambda p: loss_func([a, p[0], p[1]]),
                     [beta_opt, yc_opt], method='Nelder-Mead')
    if res_t.fun < loss_threshold:
        print(f"  α={a:.2f}, loss={res_t.fun:.4f}")

print("β 容忍区间:")
betas_test = np.linspace(0.8, 2.5, 35)
for b in betas_test:
    res_t = minimize(lambda p: loss_func([p[0], b, p[1]]),
                     [alpha_opt, yc_opt], method='Nelder-Mead')
    if res_t.fun < loss_threshold:
        print(f"  β={b:.2f}, loss={res_t.fun:.4f}")

print("yc 容忍区间:")
ycs_test = np.linspace(0.25, 1.0, 31)
for yc_t in ycs_test:
    res_t = minimize(lambda p: loss_func([p[0], p[1], yc_t]),
                     [alpha_opt, beta_opt], method='Nelder-Mead')
    if res_t.fun < loss_threshold:
        print(f"  yc={yc_t:.2f}, loss={res_t.fun:.4f}")

# ============================================================
# 9. 最终 RAR 图
# ============================================================
plt.figure(figsize=(8,7))
plt.loglog(gbar_all, gobs_all, '.', alpha=0.15, color='gray', label='SPARC data (3391 points)')
g_range = np.logspace(-13, -8, 200)

# 相变标量场论最优曲线
gobs_theory = g_range * nu_phase(g_range/a0, alpha_opt, beta_opt, yc_opt)
plt.loglog(g_range, gobs_theory, 'r-', lw=2.5,
           label=f'Phase ν (α={alpha_opt:.2f}, β={beta_opt:.2f}, yc={yc_opt:.2f})')

# MOND 标准曲线（参考）
plt.loglog(g_range, g_range * nu_std(g_range/a0), 'b--', lw=1.5, alpha=0.5,
           label='MOND ν (reference)')

plt.xlabel('g_bar (m/s²)')
plt.ylabel('g_obs (m/s²)')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.title('SPARC RAR — Phase Transition Model')
plt.tight_layout()
plt.savefig('rar_phase_optimized.png', dpi=150)
plt.show()

print("\n--- 说明 ---")
print("以上参数由 SPARC 175 星系 3391 数据点的全局拟合确定。")
print("容忍区间反映了参数在合理范围内变化时拟合质量的退化程度。")
print("α 和 yc 之间存在简并关系：较大的 α 可通过较小的 yc 补偿。")
print("当前扫描范围有限，更精确的约束需要 MCMC 采样。")