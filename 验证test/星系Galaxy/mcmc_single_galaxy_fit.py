import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. 读取 SPARC 旋转曲线数据
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
# 2. 常数与你的相变 ν 函数
# ============================================================
a0 = 1.2e-10                  # m/s²
kpc_to_m = 3.085677581e19     # m/kpc
km_to_m = 1e3                 # m/km

def nu_phase(y, alpha=3.01, beta=1.54, yc=0.48):
    """相变标量场论 ν 函数"""
    return 1.0 + alpha / (1.0 + (y / yc)**beta)

# ============================================================
# 3. MCMC 拟合单个星系
# ============================================================
def mcmc_single_galaxy(gal_name, n_steps=10000, burnin=2000):
    """
    对单个星系执行 MCMC 拟合，返回最佳参数、后验样本和诊断信息。
    """
    gal = data[data['Name'] == gal_name]
    if len(gal) == 0:
        return None

    r = gal['Rad'].values          # kpc
    Vobs = gal['Vobs'].values      # km/s
    eVobs = gal['e_Vobs'].values   # km/s
    Vgas = gal['Vgas'].values
    Vdisk = gal['Vdisk'].values
    Vbul = gal['Vbul'].values

    # 将极小误差替换为合理下限
    eVobs = np.where(eVobs < 2.0, 2.0, eVobs)

    # 判断是否有核球
    has_bulge = np.max(Vbul) > 0.1

    # ---- 对数后验函数 ----
    def log_posterior(params):
        if has_bulge:
            ml_d, ml_b = params
        else:
            ml_d, ml_b = params[0], 0.0

        # 先验：均匀分布，合理物理范围
        if ml_d <= 0.01 or ml_d > 5.0:
            return -1e10
        if has_bulge and (ml_b <= 0.01 or ml_b > 5.0):
            return -1e10

        # 计算理论速度
        Vbar2 = Vgas**2 + ml_d * Vdisk**2 + ml_b * Vbul**2
        Vbar2 = np.maximum(Vbar2, 0)
        gbar = Vbar2 / (r * kpc_to_m) * km_to_m**2
        y = gbar / a0
        nu_val = nu_phase(y)
        gobs_pred = gbar * nu_val
        Vpred = np.sqrt(gobs_pred * r * kpc_to_m) / km_to_m

        # 高斯似然
        log_like = -0.5 * np.sum(((Vobs - Vpred) / eVobs)**2)
        return log_like

    # ---- 先用快速拟合找初始点 ----
    def chi2_quick(params):
        if has_bulge:
            ml_d, ml_b = params
        else:
            ml_d, ml_b = params[0], 0.0
        Vbar2 = Vgas**2 + ml_d * Vdisk**2 + ml_b * Vbul**2
        Vbar2 = np.maximum(Vbar2, 0)
        gbar = Vbar2 / (r * kpc_to_m) * km_to_m**2
        y = gbar / a0
        gobs_pred = gbar * nu_phase(y)
        Vpred = np.sqrt(gobs_pred * r * kpc_to_m) / km_to_m
        return np.sum(((Vobs - Vpred) / eVobs)**2)

    if has_bulge:
        res_quick = minimize(chi2_quick, x0=[0.5, 0.7], bounds=[(0.01, 5), (0.01, 5)])
        p0 = res_quick.x
        n_params = 2
    else:
        res_quick = minimize(lambda p: chi2_quick([p[0]]), x0=[0.5], bounds=[(0.01, 5)])
        p0 = [res_quick.x[0]]
        n_params = 1

    # ---- Metropolis-Hastings MCMC ----
    chain = np.zeros((n_steps, n_params))
    chain[0] = p0
    current_lp = log_posterior(p0)
    accepted = 0

    # 自适应步长
    step_scale = 0.1
    for i in range(1, n_steps):
        proposal = chain[i-1] + step_scale * np.random.randn(n_params) * np.array(p0)
        proposal_lp = log_posterior(proposal)
        if np.log(np.random.rand()) < proposal_lp - current_lp:
            chain[i] = proposal
            current_lp = proposal_lp
            accepted += 1
        else:
            chain[i] = chain[i-1]

    # 后验样本（丢弃燃烧期）
    samples = chain[burnin:]
    if has_bulge:
        ml_d_med = np.median(samples[:,0])
        ml_b_med = np.median(samples[:,1])
        ml_d_err = np.std(samples[:,0])
        ml_b_err = np.std(samples[:,1])
    else:
        ml_d_med = np.median(samples[:,0])
        ml_b_med = 0.0
        ml_d_err = np.std(samples[:,0])
        ml_b_err = 0.0

    # 最佳拟合的速度曲线
    Vbar2_best = Vgas**2 + ml_d_med * Vdisk**2 + ml_b_med * Vbul**2
    Vbar2_best = np.maximum(Vbar2_best, 0)
    gbar_best = Vbar2_best / (r * kpc_to_m) * km_to_m**2
    y_best = gbar_best / a0
    gobs_best = gbar_best * nu_phase(y_best)
    Vpred_best = np.sqrt(gobs_best * r * kpc_to_m) / km_to_m

    # 残差与 χ²
    residual = (Vobs - Vpred_best) / Vobs
    chi2 = np.sum(((Vobs - Vpred_best) / eVobs)**2)
    dof = len(r) - n_params
    reduced_chi2 = chi2 / dof if dof > 0 else np.inf

    return {
        'name': gal_name,
        'ml_disk': ml_d_med,
        'ml_bulge': ml_b_med,
        'ml_disk_err': ml_d_err,
        'ml_bulge_err': ml_b_err,
        'has_bulge': has_bulge,
        'chi2': chi2,
        'dof': dof,
        'reduced_chi2': reduced_chi2,
        'acceptance': accepted / n_steps,
        'samples': samples,
        'r': r, 'Vobs': Vobs, 'eVobs': eVobs,
        'Vpred': Vpred_best,
        'residual': residual
    }

# ============================================================
# 4. 批量拟合所有星系
# ============================================================
print("开始 MCMC 批量拟合...")
results = []
galaxies = data['Name'].unique()

for i, name in enumerate(galaxies):
    res = mcmc_single_galaxy(name, n_steps=10000, burnin=2000)
    if res is not None:
        results.append(res)
        if (i+1) % 10 == 0:
            print(f"已完成 {i+1}/{len(galaxies)} 个星系")

# ============================================================
# 5. 汇总统计
# ============================================================
print(f"\n成功拟合 {len(results)} 个星系")
chi2_list = [r['reduced_chi2'] for r in results if r['reduced_chi2'] < 100]
print(f"reduced χ² 中位数: {np.median(chi2_list):.2f}")
print(f"reduced χ² < 3 的比例: {sum(1 for c in chi2_list if c < 3) / len(chi2_list) * 100:.1f}%")

# ============================================================
# 6. 典型星系绘图
# ============================================================
for name in ['NGC2403', 'NGC2841', 'DDO154']:
    match = [r for r in results if r['name'] == name]
    if not match:
        continue
    r = match[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
    ax1.errorbar(r['r'], r['Vobs'], yerr=r['eVobs'], fmt='o', capsize=3, label='Observed')
    ax1.plot(r['r'], r['Vpred'], 'r-', lw=2, label='Phase-transition model')
    ax1.set_xlabel('Radius (kpc)')
    ax1.set_ylabel('V (km/s)')
    ax1.set_title(f'{name}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.errorbar(r['r'], r['residual'], yerr=r['eVobs']/r['Vobs'], fmt='o', capsize=3)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('(V_obs - V_model)/V_obs')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{name}_mcmc.png', dpi=150)
    plt.show()

print("MCMC 拟合完成。")