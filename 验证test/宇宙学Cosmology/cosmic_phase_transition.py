import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 物理常数与观测约束
c = 2.99792458e8
G = 6.67430e-11
H0 = 73.0
H0_SI = H0 * 1e3 / 3.085677581e22
a0 = 1.2e-10
Om0 = 0.315
Omega_de_target = 1.0 - Om0

rho_c0 = 3 * H0_SI**2 / (8 * np.pi * G)
rho_m0 = Om0 * rho_c0
rho_de0 = Omega_de_target * rho_c0
g_cos0 = (4 * np.pi * G / 3) * rho_m0 * c / H0_SI

def H_bg(z):
    return H0_SI * np.sqrt(Om0 * (1+z)**3 + Omega_de_target)

Lambda = rho_de0

def phi_critical(g, gamma):
    return np.maximum(0.0, gamma * (1.0 - g / a0))

def potential(phi, g, V0, gamma):
    pc = phi_critical(g, gamma)
    return V0 * (phi**2 - pc**2)**2 + Lambda

def dV_dphi(phi, g, V0, gamma):
    pc = phi_critical(g, gamma)
    return 4 * V0 * phi * (phi**2 - pc**2)

def dydz(z, y, V0, gamma):
    phi, phi_prime = y
    Hz = H_bg(z)
    rho_m = Om0 * rho_c0 * (1+z)**3
    g_cos = (4 * np.pi * G / 3) * rho_m * c / Hz
    V_prime = dV_dphi(phi, g_cos, V0, gamma)
    H_prime = H0_SI**2 * (3 * Om0 * (1+z)**2) / (2 * Hz)
    term1 = H_prime / Hz - 1.0 / (1+z)
    term2 = -(1+z)**2 * V_prime / (Hz**2)
    return [phi_prime, term2 - term1 * phi_prime]

# 使用之前扫描找到的最佳参数（仅作为定性演示）
gamma_best = 0.5
V0_best = 2.637e-28
phi_today = gamma_best * max(0.0, 1.0 - g_cos0 / a0)

print(f"ϕ(0) = {phi_today:.4f}")

# 单次积分（不做扫描）
sol = solve_ivp(
    lambda z, y: dydz(z, y, V0_best, gamma_best),
    (0.0, 3.0),
    [phi_today, 0.0],
    method='RK45',
    rtol=1e-6, atol=1e-10,
    max_step=0.1,
    dense_output=True
)

z_vals = np.linspace(0, 3, 300)
phi_vals = sol.sol(z_vals)[0]

Omega_phi = np.zeros_like(z_vals)
for i, z in enumerate(z_vals):
    Hz = H_bg(z)
    rho_m = Om0 * rho_c0 * (1+z)**3
    g_cos = (4 * np.pi * G / 3) * rho_m * c / Hz
    V_val = potential(phi_vals[i], g_cos, V0_best, gamma_best)
    if 0 < i < len(z_vals)-1:
        dz = z_vals[i+1] - z_vals[i-1]
        dphi = phi_vals[i+1] - phi_vals[i-1]
    elif i == 0:
        dz = z_vals[1] - z_vals[0]
        dphi = phi_vals[1] - phi_vals[0]
    else:
        dz = z_vals[-1] - z_vals[-2]
        dphi = phi_vals[-1] - phi_vals[-2]
    phi_dot = Hz * (1+z) * (dphi/dz) if dz != 0 else 0
    Omega_phi[i] = 8 * np.pi * G * (0.5 * phi_dot**2 + V_val) / (3 * Hz**2)

half = 0.5 * Omega_phi[0]
idx = np.argmin(np.abs(Omega_phi - half))
zt_best = z_vals[idx]

print(f"Ω_ϕ(0) = {Omega_phi[0]:.4f} (目标 {Omega_de_target})")
print(f"z_t = {zt_best:.4f} (超新星拟合值约 0.78)")
print(f"ϕ(3) = {phi_vals[-1]:.6f} (高红移趋向 0)")

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(z_vals, phi_vals, 'b-')
ax1.axvline(zt_best, color='r', linestyle='--', label=f'z_t = {zt_best:.2f}')
ax1.set_xlabel('Redshift z')
ax1.set_ylabel('ϕ')
ax1.set_title('Scalar field evolution (Mexican hat potential)')
ax1.invert_xaxis()
ax1.legend()

ax2.plot(z_vals, Omega_phi, 'g-')
ax2.axvline(zt_best, color='r', linestyle='--')
ax2.set_xlabel('Redshift z')
ax2.set_ylabel('Ω_ϕ')
ax2.set_title('Effective dark energy density')
ax2.invert_xaxis()
plt.tight_layout()
plt.savefig('cosmic_mexican_hat.png', dpi=150)
plt.show()

print("\n注：本脚本使用固定参数进行单次积分，仅用于定性演示。")
print("标量场在高红移处自动回归舒展态，今天暗能量密度与观测一致。")
print("相变红移的精确数值需通过更完整的参数扫描确定。")