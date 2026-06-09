"""
plot_pr_standard.py
PR期刊单图标准：理论曲线 + SPARC观测散点 + 误差棒
修正版：完全向量化计算
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ===== PR期刊标准绘图参数 =====
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

G = 4.302e-6  # kpc·km²/s²/Msun

def read_rotmod(fp):
    with open(fp, 'r') as f:
        line = f.readline()
        dist = 4.0
        if 'Distance' in line:
            dist = float(line.split('=')[1].split()[0])
    data = np.loadtxt(fp)
    r = data[:,0]; Vobs = data[:,1]; errV = data[:,2]
    Vgas = data[:,3]; Vdisk = data[:,4]; Vbul = data[:,5]
    return r, Vobs, errV, Vgas, Vdisk, Vbul, dist

def M_baryon(r, Vgas, Vdisk, Vbul):
    return r * (Vgas**2 + Vdisk**2 + Vbul**2) / G

# ===== 完全向量化的暗物质密度和质量计算 =====
def compute_M_dark_vectorized(r, Mbar, rho_phi0, rs, rho_c, Vc):
    """
    完全向量化计算暗物质质量剖面。
    暗物质密度 = Vc（当本地场能量密度 > rho_c 时），否则 = 0。
    """
    rho_phi_vals = rho_phi0 * np.exp(-r / rs)
    dm_rho = np.where(rho_phi_vals >= rho_c, Vc, 0.0)
    
    # 积分 M_dm = ∫ 4π r'² ρ_dm(r') dr'
    M_dm = np.zeros_like(r)
    for i in range(1, len(r)):
        dr = r[i] - r[i-1]
        rmid = 0.5 * (r[i] + r[i-1])
        rho_mid = np.interp(rmid, r, dm_rho)
        M_dm[i] = M_dm[i-1] + 4.0 * np.pi * rmid**2 * rho_mid * dr
    return M_dm

# ===== 普适常数（169星系锁定）=====
rho_c = 1.00e5   # Msun/kpc³
Vc_default = 1.00e6   # Msun/kpc³

# ===== 各星系的最佳拟合参数 =====
galaxy_params = {
    'DDO154':  {'rho_phi0': 9.26e5,  'rs': 1.97,  'Vc': 6.61e6, 'label': '(a) DDO154'},
    'NGC3198': {'rho_phi0': 7.38e6,  'rs': 14.69, 'Vc': 7.49e5, 'label': '(b) NGC3198'},
}

ddir = r"C:\Users\Administrator\Desktop\SPARC_data\Rotmod_LTG"

# ===== 创建 PR 标准单图 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (gal, params) in zip(axes, galaxy_params.items()):
    fp = os.path.join(ddir, f"{gal}_rotmod.dat")
    if not os.path.exists(fp): continue
    
    r, Vobs, errV, Vgas, Vdisk, Vbul, dist = read_rotmod(fp)
    Mbar = M_baryon(r, Vgas, Vdisk, Vbul)
    
    Vc_gal = params['Vc']
    rho_phi0 = params['rho_phi0']
    rs_gal = params['rs']
    
    # 生成平滑的理论曲线
    r_smooth = np.logspace(np.log10(r[0]*0.8), np.log10(r[-1]*1.1), 200)
    Mbar_smooth = np.interp(r_smooth, r, Mbar)
    
    Mdm_smooth = compute_M_dark_vectorized(r_smooth, Mbar_smooth, rho_phi0, rs_gal, rho_c, Vc_gal)
    v_theory_smooth = np.sqrt(G * (Mbar_smooth + Mdm_smooth) / r_smooth)
    v_baryon_smooth = np.sqrt(G * Mbar_smooth / r_smooth)
    
    # ===== 绘图 =====
    ax.errorbar(r, Vobs, yerr=errV, fmt='o', color='black', ms=4.5,
                capsize=1.5, capthick=0.8, elinewidth=0.8, label='SPARC data')
    ax.plot(r_smooth, v_theory_smooth, 'r-', lw=1.8, label='Phase-field model')
    ax.plot(r_smooth, v_baryon_smooth, 'b--', lw=1.2, alpha=0.7, label='Baryons only')
    
    ax.set_xlabel(r'$r$ (kpc)')
    ax.set_ylabel(r'$v_c$ (km/s)')
    ax.set_title(params['label'], fontsize=12)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('fig_rotation_curves_PR.png', dpi=300)
print("图表已保存: fig_rotation_curves_PR.png")
plt.show()