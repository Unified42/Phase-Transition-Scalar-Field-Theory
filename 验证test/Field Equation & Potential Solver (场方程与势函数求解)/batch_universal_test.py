"""
batch_universal_test.py
全星系检验：相态势能 + 相变张力双参数模型的普适性
"""

import numpy as np
from scipy.optimize import minimize
import os
import glob

G = 4.302e-6

def read_rotmod(fp):
    try:
        with open(fp, 'r') as f:
            line = f.readline()
            dist = 4.0
            if 'Distance' in line:
                dist = float(line.split('=')[1].split()[0])
        data = np.loadtxt(fp, ndmin=2)
        if data.shape[0] < 5: return None
        r = data[:,0]; Vobs = data[:,1]; errV = data[:,2]
        Vgas = data[:,3]; Vdisk = data[:,4]; Vbul = data[:,5]
        return r, Vobs, errV, Vgas, Vdisk, Vbul, dist
    except:
        return None

def M_baryon(r, Vgas, Vdisk, Vbul):
    return r * (Vgas**2 + Vdisk**2 + Vbul**2) / G

def rho_phi_func(r, rho_phi0, rs):
    return rho_phi0 * np.exp(-r / rs)

def tension_func(r, rho_phi0, rs, alpha):
    rho = rho_phi_func(r, rho_phi0, rs)
    drho = np.abs(np.gradient(rho, r))
    return alpha * drho * r

def get_rho_DM(r, rho_phi0, rs, rho_c, T_c, Vc, alpha):
    rho_phi_vals = rho_phi_func(r, rho_phi0, rs)
    T_vals = tension_func(r, rho_phi0, rs, alpha)
    dm_rho = np.zeros_like(r)
    for i in range(len(r)):
        if rho_phi_vals[i] >= rho_c or T_vals[i] >= T_c:
            dm_rho[i] = Vc
        else:
            dm_rho[i] = 0.0
    return dm_rho

def M_dark(r, rho_phi0, rs, rho_c, T_c, Vc, alpha):
    dm_rho_vals = get_rho_DM(r, rho_phi0, rs, rho_c, T_c, Vc, alpha)
    M = np.zeros_like(r)
    for i in range(1, len(r)):
        dr = r[i] - r[i-1]
        rmid = 0.5 * (r[i] + r[i-1])
        rho_mid = np.interp(rmid, r, dm_rho_vals)
        M[i] = M[i-1] + 4.0 * np.pi * rmid**2 * rho_mid * dr
    return M

def chi2(params, r, Vobs, errV, Mbar):
    rho_phi0, rs, rho_c, T_c, Vc, alpha = params
    if any(p <= 0 for p in [rho_phi0, rs, rho_c, T_c, Vc, alpha]):
        return 1e20
    Mdm = M_dark(r, rho_phi0, rs, rho_c, T_c, Vc, alpha)
    v_total_sq = G * (Mbar + Mdm) / r
    v_pred = np.sqrt(np.clip(v_total_sq, 0, None))
    return np.sum(((v_pred - Vobs) / errV)**2)

# 数据目录
ddir = r"C:\Users\Administrator\Desktop\SPARC_data\Rotmod_LTG"
files = glob.glob(os.path.join(ddir, "*_rotmod.dat"))
print(f"找到 {len(files)} 个星系文件。")

results = []

for fp in files:
    galname = os.path.splitext(os.path.basename(fp))[0].replace("_rotmod", "")
    data = read_rotmod(fp)
    if data is None: continue
    r, Vobs, errV, Vgas, Vdisk, Vbul, dist = data
    Mbar = M_baryon(r, Vgas, Vdisk, Vbul)

    p0 = [1e7, r[-1]/2, 1e5, 1e4, 1.0e6, 0.1]
    bounds = [(1e4, 1e9), (0.5, 100.0), (1e3, 1e8), (1e1, 1e7), (1e5, 5e7), (1e-4, 10.0)]

    try:
        res = minimize(chi2, p0, args=(r, Vobs, errV, Mbar), bounds=bounds, method='L-BFGS-B')
        if res.success:
            p = res.x
            Mdm = M_dark(r, p[0], p[1], p[2], p[3], p[4], p[5])
            v_pred = np.sqrt(G * (Mbar + Mdm) / r)
            rmse = np.sqrt(np.mean((v_pred - Vobs)**2))
            v_baryon = np.sqrt(G * Mbar / r)
            rmse_baryon = np.sqrt(np.mean((v_baryon - Vobs)**2))
            improvement = (rmse_baryon - rmse) / rmse_baryon * 100

            rho_dm_final = get_rho_DM(r, p[0], p[1], p[2], p[3], p[4], p[5])
            r_trans = r[rho_dm_final == 0][0] if np.any(rho_dm_final == 0) else r[-1]
            ratio = Mdm[-1] / Mbar[-1] if Mbar[-1] > 0 else 0

            results.append({
                'galaxy': galname,
                'Vc': p[4], 'rho_c': p[2], 'T_c': p[3],
                'r_trans': r_trans, 'rmse': rmse, 'improvement': improvement,
                'ratio': ratio
            })
    except:
        pass

print(f"\n成功拟合 {len(results)} 个星系。\n")
print(f"{'星系':<15} {'Vc':<12} {'rho_c':<12} {'T_c':<12} {'r_trans':<10} {'RMSE':<8} {'改善':<8}")
print("-" * 80)
for r in results:
    print(f"{r['galaxy']:<15} {r['Vc']:<12.2e} {r['rho_c']:<12.2e} {r['T_c']:<12.2e} {r['r_trans']:<10.2f} {r['rmse']:<8.2f} {r['improvement']:<8.1f}%")

# 统计普适性
rho_c_vals = np.array([r['rho_c'] for r in results])
T_c_vals = np.array([r['T_c'] for r in results])
Vc_vals = np.array([r['Vc'] for r in results])

print(f"\n===== 普适性统计 =====")
print(f"rho_c: 中位数={np.median(rho_c_vals):.2e}, 散射={np.std(np.log10(rho_c_vals)):.2f} dex")
print(f"T_c:   中位数={np.median(T_c_vals):.2e}, 散射={np.std(np.log10(T_c_vals)):.2f} dex")
print(f"V_c:   中位数={np.median(Vc_vals):.2e}, 散射={np.std(np.log10(Vc_vals)):.2f} dex")

# 改善率分布
improvements = np.array([r['improvement'] for r in results])
n_positive = np.sum(improvements > 10)
print(f"\n改善率 > 10% 的星系: {n_positive}/{len(results)}")
print(f"改善率中位数: {np.median(improvements):.1f}%")