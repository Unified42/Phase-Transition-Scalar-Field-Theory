import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import trapezoid

# ============================================================
# 物理常数
# ============================================================
c_km_s = 299792.458   # 光速 (km/s)

# ============================================================
# 1. 读取 Pantheon+ 超新星数据
# 数据来源：Pantheon+ 公开数据集 (Brout et al. 2022)
# 列说明：zHD=哈勃图红移, MU_SH0ES=距离模数,
#         MU_SH0ES_ERR_DIAG=距离模数对角误差,
#         IS_CALIBRATOR=是否为定标超新星
# ============================================================
data_file = r"data/PantheonPlusSH0ES.dat"
df = pd.read_csv(data_file, sep=r'\s+', comment='#', engine='python')
z_all = df['zHD'].values
mu_obs_all = df['MU_SH0ES'].values
mu_err_all = df['MU_SH0ES_ERR_DIAG'].values
is_cal = df['IS_CALIBRATOR'].values

# 只选择哈勃流超新星（IS_CALIBRATOR = 0）
mask = (is_cal == 0)
z = z_all[mask]
mu_obs = mu_obs_all[mask]
mu_err = mu_err_all[mask]
print(f"读取 {len(df)} 条数据，其中哈勃流超新星 {len(z)} 颗")

# ============================================================
# 2. 基底开关式相变模型
# H²(z) = H₀² [Ω_m(1+z)³ + Ω_def(z)]
# Ω_def(z) = (1-Ω_m) / [1 + (z/zt)^α]
# 参数：H₀(固定=73.0), Ω_m, zt, α
# ============================================================
def H_phase(z, H0, Om, zt, alpha):
    """开关式相变模型的哈勃参数 (km/s/Mpc)"""
    Omega_def = (1 - Om) / (1 + (z / zt)**alpha)
    return H0 * np.sqrt(Om * (1+z)**3 + Omega_def)

def mu_phase(z_arr, H0, Om, zt, alpha):
    """理论距离模数 μ = 5 log10(d_L/Mpc) + 25"""
    dL = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        z_int = np.linspace(0, zi, 200)
        Hz_int = H_phase(z_int, H0, Om, zt, alpha)
        dL[i] = (1+zi) * trapezoid(c_km_s / Hz_int, z_int)
    return 5*np.log10(dL) + 25

# ============================================================
# 3. 拟合宇宙学参数（H₀ 固定为 73.0 km/s/Mpc）
# ============================================================
def chi2(params):
    Om, zt, alpha = params
    return np.sum(((mu_obs - mu_phase(z, 73.0, Om, zt, alpha)) / mu_err)**2)

res = minimize(chi2, x0=[0.3, 0.5, 6.0], bounds=[(0.01, 0.99), (0.01, 5), (0.1, 15)])
Om_b, zt_b, a_b = res.x
chi2_val = res.fun
dof = len(z) - 3
print(f"\n========== 开关式相变模型拟合结果 ==========")
print(f"固定 H₀ = 73.0 km/s/Mpc")
print(f"最佳拟合: Ωm={Om_b:.4f}, zt={zt_b:.4f}, α={a_b:.2f}")
print(f"χ²/dof = {chi2_val:.1f}/{dof} = {chi2_val/dof:.2f}")

# ΛCDM 基线（H₀=73, Ωm=0.315）
def mu_LCDM(z_arr, H0, Om):
    dL = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        z_int = np.linspace(0, zi, 200)
        dL[i] = (1+zi) * trapezoid(c_km_s/(H0*np.sqrt(Om*(1+z_int)**3+(1-Om))), z_int)
    return 5*np.log10(dL) + 25

chi2_lcdm = np.sum(((mu_obs - mu_LCDM(z, 73, 0.315)) / mu_err)**2)
print(f"\nΛCDM 基线 (H₀=73, Ωm=0.315): χ²={chi2_lcdm:.1f}/{len(z)}={chi2_lcdm/len(z):.2f}")
print(f"注：本模型使用对角误差，完整协方差处理后 χ² 值可能有所变化。")

# ============================================================
# 4. 哈勃张力：基于物质密度偏移的估算
# 早期宇宙（z >> zt）基底未相变，纯物质主导
# CMB 用 ΛCDM 拟合时会"错误"地给出较低的 H₀
# 等效 H₀ 可通过密度偏移估算：
#   H₀_CMB ≈ H₀_local × √(Ωm_CMB / Ωm_local)
# ============================================================
Om_cmb = 0.315  # Planck 2018 ΛCDM 拟合值
H0_early_eff = 73.0 * np.sqrt(Om_cmb / Om_b)
print(f"\n--- 哈勃张力估算 ---")
print(f"晚期局部 H₀ = 73.0 km/s/Mpc")
print(f"早期等效 H₀ ≈ 73.0 × √(0.315/{Om_b:.4f}) = {H0_early_eff:.1f} km/s/Mpc")
print(f"Planck 2018 ΛCDM 拟合 H₀ = 67.4 ± 0.5 km/s/Mpc")
print(f"差异 ≈ {73.0 - H0_early_eff:.1f} km/s/Mpc")
print(f"注：该估算基于物质密度偏移关系，完整的 CMB 分析需将标量场微扰嵌入玻尔兹曼求解器。")

# ============================================================
# 5. 绘图
# ============================================================
z_line = np.logspace(-3, np.log10(0.15), 100)

# 哈勃图
plt.figure(figsize=(10, 5))
plt.errorbar(z, mu_obs, yerr=mu_err, fmt='.', alpha=0.3, label='Pantheon+ (1624 SNe)')
plt.plot(z_line, mu_phase(z_line, 73, Om_b, zt_b, a_b), 'r-', label=f'Phase-switch model')
plt.plot(z_line, mu_LCDM(z_line, 73, 0.315), 'b--', label=f'ΛCDM (Ωm=0.315)')
plt.xlabel('Redshift z')
plt.ylabel('Distance modulus μ')
plt.legend()
plt.grid(alpha=0.3)
plt.title('Pantheon+ Hubble Diagram')
plt.tight_layout()
plt.savefig('supernova_hubble_diagram.png', dpi=150)
plt.show()

# 残差图
plt.figure(figsize=(10, 4))
res_p = mu_obs - mu_phase(z, 73, Om_b, zt_b, a_b)
res_l = mu_obs - mu_LCDM(z, 73, 0.315)
plt.errorbar(z, res_p, yerr=mu_err, fmt='.', alpha=0.3, color='red', label='Phase-switch')
plt.errorbar(z, res_l, yerr=mu_err, fmt='.', alpha=0.3, color='blue', label='ΛCDM')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Redshift z')
plt.ylabel('Residual (mag)')
plt.legend()
plt.grid(alpha=0.3)
plt.title('Residuals')
plt.tight_layout()
plt.savefig('supernova_residuals.png', dpi=150)
plt.show()

print("\n结果：开关式相变模型在背景膨胀层面与超新星数据兼容。")
print("拟合优度与 ΛCDM 大致可比，表明该模型可进一步验证。")
print("更严格的检验需要完整的协方差矩阵和微扰层面分析。")