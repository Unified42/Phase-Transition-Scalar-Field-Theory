import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================
# 1. 读取 SPARC 旋转曲线数据
# 数据来源：SPARC 数据库 (Lelli et al. 2016)
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
# 2. 物理常数与相变 ν 函数
# ============================================================
a0 = 1.2e-10                  # m/s²，临界加速度
kpc_to_m = 3.085677581e19     # m/kpc
km_to_m = 1e3                 # m/km

def nu_phase(y, alpha=3.01, beta=1.54, yc=0.48):
    """相变标量场论 ν 函数"""
    return 1.0 + alpha / (1.0 + (y / yc)**beta)

# ============================================================
# 3. 用标准 MOND ν 函数初步拟合质光比
# 目的：获得 g_bar 和 g_obs 的初始估计
# ============================================================
def nu_std(y):
    return 0.5 + 0.5 * np.sqrt(1 + 4.0 / y)

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
print(f"有效数据点: {len(data)}")

# ============================================================
# 4. 网格搜索最优参数
# 在 α、β、yc 的参数空间中进行离散搜索
# 目标：最小化 |med-1| + scatter
# ============================================================
alphas = np.arange(1, 11, 1)      # α 候选值 1-10
betas = np.arange(1, 7, 0.5)      # β 候选值 1-6
ycs = np.arange(0.5, 2.1, 0.25)   # yc 候选值 0.5-2.0

best_params = None
best_metric = 1e9
results_log = []

for alpha in alphas:
    for beta in betas:
        for yc in ycs:
            y = data['gbar'].values / a0
            nu = nu_phase(y, alpha, beta, yc)
            gobs_pred = data['gbar'].values * nu
            ratio = data['gobs'].values / gobs_pred
            med = np.median(ratio)
            scatter = np.std(np.log10(ratio))
            loss = abs(med - 1.0) + scatter
            results_log.append((alpha, beta, yc, med, scatter, loss))
            if loss < best_metric:
                best_metric = loss
                best_params = (alpha, beta, yc)

alpha_best, beta_best, yc_best = best_params

print(f"\n网格搜索完成，总组合数: {len(results_log)}")
print(f"最佳参数: α={alpha_best}, β={beta_best}, yc={yc_best}")
print(f"最佳指标 (|med-1|+scatter) = {best_metric:.4f}")

# 最终统计
y_all = data['gbar'].values / a0
nu_best = nu_phase(y_all, alpha_best, beta_best, yc_best)
gobs_pred_best = data['gbar'].values * nu_best
ratio_best = data['gobs'].values / gobs_pred_best
med_best = np.median(ratio_best)
scat_best = np.std(np.log10(ratio_best))
print(f"中位数: {med_best:.3f}, 散射: {scat_best:.3f} dex")

# ============================================================
# 5. 网格搜索 RAR 图
# ============================================================
plt.figure(figsize=(8,7))
plt.loglog(data['gbar'], data['gobs'], '.', alpha=0.2, color='gray', label='SPARC data')
g_range = np.logspace(-13, -8, 200)
gobs_theory_line = g_range * nu_phase(g_range/a0, alpha_best, beta_best, yc_best)
plt.loglog(g_range, gobs_theory_line, 'r-', lw=2.5,
           label=f'Phase ν (α={alpha_best}, β={beta_best}, yc={yc_best})')
plt.xlabel('g_bar (m/s²)')
plt.ylabel('g_obs (m/s²)')
plt.legend()
plt.grid(alpha=0.3, which='both')
plt.title('SPARC RAR — Grid Search Result')
plt.tight_layout()
plt.savefig('rar_phase_grid.png', dpi=150)
plt.show()

print("\n注意：本脚本使用离散网格搜索，分辨率有限。")
print("更精确的参数值和容忍区间见 phase_nu_profile.py 的精细扫描结果。")