import numpy as np
from scipy.optimize import minimize
import os

# 物理常数
G = 4.302e-6  # (km/s)^2 * kpc / Msun

def read_rotmod(fp):
    """读取SPARC数据"""
    with open(fp, 'r') as f:
        line = f.readline()
        dist = 4.0
        if 'Distance' in line:
            dist = float(line.split('=')[1].split()[0])
    data = np.loadtxt(fp)
    # r, Vobs, errV, Vgas, Vdisk, Vbul
    return data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], dist

def M_baryon(r, Vgas, Vdisk, Vbul):
    """计算总重子质量分布"""
    return r * (Vgas**2 + Vdisk**2 + Vbul**2) / G

def rho_phi_func(r, rho_phi0, rs):
    """场相态势能：描述场的基础能量分布"""
    return rho_phi0 * np.exp(-r / rs)

def tension_func(r, rho_phi0, rs, alpha):
    """
    相变张力：T = alpha * |drho/dr| * r
    物理意义：形变产生的张力随半径（扩张效应）放大。
    """
    rho = rho_phi_func(r, rho_phi0, rs)
    # 计算梯度
    drho = np.abs(np.gradient(rho, r))
    return alpha * drho * r

def get_rho_DM(r, rho_phi0, rs, rho_c, T_c, Vc, alpha):
    """
    双参数相变逻辑：
    场在内部维持蜷缩态(Vc)，直到势能和张力跌破阈值，‘崩断’进入准舒展态(0)。
    """
    rho_phi_vals = rho_phi_func(r, rho_phi0, rs)
    T_vals = tension_func(r, rho_phi0, rs, alpha)
    
    dm_rho = np.zeros_like(r)
    
    # 从内向外扫描：蜷缩态由于其‘刚性’，在内部维持稳定
    # 只要满足维持条件之一，就锁定在蜷缩态 Vc
    for i in range(len(r)):
        # 只要势能足够高，或者张力尚能支撑构型
        if rho_phi_vals[i] >= rho_c or T_vals[i] >= T_c:
            dm_rho[i] = Vc
        else:
            # 跌落至准舒展态
            dm_rho[i] = 0.0
            
    return dm_rho

def M_dark(r, rho_phi0, rs, rho_c, T_c, Vc, alpha):
    """计算积分暗物质质量"""
    dm_rho_vals = get_rho_DM(r, rho_phi0, rs, rho_c, T_c, Vc, alpha)
    M = np.zeros_like(r)
    for i in range(1, len(r)):
        dr = r[i] - r[i-1]
        rmid = 0.5 * (r[i] + r[i-1])
        # 插值获取中点密度
        rho_mid = np.interp(rmid, r, dm_rho_vals)
        M[i] = M[i-1] + 4.0 * np.pi * rmid**2 * rho_mid * dr
    return M

def chi2(params, r, Vobs, errV, Mbar):
    rho_phi0, rs, rho_c, T_c, Vc, alpha = params
    # 物理边界约束
    if any(p <= 0 for p in [rho_phi0, rs, rho_c, T_c, Vc, alpha]):
        return 1e20
    
    Mdm = M_dark(r, rho_phi0, rs, rho_c, T_c, Vc, alpha)
    # 总引力产生的速度 pred
    v_total_sq = G * (Mbar + Mdm) / r
    v_pred = np.sqrt(np.clip(v_total_sq, 0, None))
    
    return np.sum(((v_pred - Vobs) / errV)**2)

# --- 执行验证 ---
ddir = r"C:\Users\Administrator\Desktop\SPARC_data\Rotmod_LTG"
galaxies = ["DDO154", "NGC2403", "NGC2841", "NGC3198", "NGC6503", "DDO161", "IC2574"]

print("=" * 80)
print("  纯场相态势能/相变张力双参数模型验证")
print("=" * 80)

for gal in galaxies:
    fp = os.path.join(ddir, f"{gal}_rotmod.dat")
    if not os.path.exists(fp): continue
    
    r, Vobs, errV, Vgas, Vdisk, Vbul, dist = read_rotmod(fp)
    Mbar = M_baryon(r, Vgas, Vdisk, Vbul)
    
    # 初始猜测：Vc 锚定到你理论中的 10^6 数量级
    p0 = [1e7, r[-1]/2, 1e5, 1e4, 1.0e6, 0.1]
    bounds = [(1e4, 1e9), (0.5, 100.0), (1e3, 1e8), (1e1, 1e7), (1e5, 5e7), (1e-4, 10.0)]
    
    res = minimize(chi2, p0, args=(r, Vobs, errV, Mbar), bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        p = res.x
        Mdm = M_dark(r, p[0], p[1], p[2], p[3], p[4], p[5])
        v_pred = np.sqrt(G * (Mbar + Mdm) / r)
        rmse = np.sqrt(np.mean((v_pred - Vobs)**2))
        
        # 寻找相变发生的边界半径
        rho_dm_final = get_rho_DM(r, p[0], p[1], p[2], p[3], p[4], p[5])
        r_trans = r[rho_dm_final == 0][0] if np.any(rho_dm_final == 0) else r[-1]
        
        print(f"星系: {gal}")
        print(f"  [锁定值] 蜷缩态密度 Vc: {p[4]:.2e} Msun/kpc^3")
        print(f"  [临界点] 势能阈值 rho_c: {p[2]:.2e}, 张力阈值 T_c: {p[3]:.2e}")
        print(f"  [相变点] 场跌落半径: {r_trans:.2f} kpc")
        print(f"  [拟合度] RMSE: {rmse:.2f} km/s")
        print("-" * 40)
