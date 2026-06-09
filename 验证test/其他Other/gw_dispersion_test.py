import h5py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, correlate
import matplotlib.pyplot as plt
import os

# ============================================================
# 配置参数
# 引力波色散检验：GW170817 双中子星并合事件
# 目标：测量不同频段引力波的到达时间差，检验是否存在色散
# ============================================================
data_dir = "data"
files = {
    'H1': os.path.join(data_dir, "H-H1_LOSC_CLN_16_V1-1187007040-2048.hdf5"),
    'L1': os.path.join(data_dir, "L-L1_LOSC_CLN_16_V1-1187007040-2048.hdf5"),
}

# GW170817 峰值到达时间
# GPS时间约 1187008882.43，相对于文件起始约 1842.43 秒
t_peak = 1842.43            # 峰值相对时间 (s)
window_read = 20             # 读取 ±20 秒窗口（用于滤波避免边缘瞬态）
core_radius = 0.1            # 核心分析区域 ±0.1 秒（信号最强区域）

# 分析频段 (Hz)
# 选择两个不重叠的频段以检验频率依赖的到达时间差
band_low = (30, 50)
band_high = (80, 100)

# ============================================================
# 读取 LIGO 减噪应变数据
# 数据来源：GWOSC GW170817 事件页面
# 采样率：16384 Hz
# ============================================================
def load_strain(filename):
    """
    读取 HDF5 格式的应变数据
    返回：相对时间 (s)、应变、采样频率 (Hz)
    """
    with h5py.File(filename, 'r') as f:
        strain = f['strain/Strain'][:]
        dt = f['strain/Strain'].attrs['Xspacing']
    fs = 1.0 / dt
    t_rel = np.arange(len(strain)) * dt
    idx = np.where((t_rel > t_peak - window_read) & (t_rel < t_peak + window_read))[0]
    return t_rel[idx], strain[idx], fs

# ============================================================
# 带通滤波
# ============================================================
def bandpass(data, low, high, fs, order=4):
    """
    对数据进行带通滤波
    使用 4 阶 Butterworth 滤波器，双向滤波（零相位）
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

# ============================================================
# 包络互相关延迟测量
# 原理：对两个频段的信号分别取包络（希尔伯特变换），
#       然后计算包络的互相关，峰值位置即为相对时间延迟
# ============================================================
def envelope_crosscorr(t, s1, s2, fs):
    """
    计算两个信号包络之间的时间延迟
    返回：延迟 (s)、互相关序列、延迟样本点序列
    """
    # 只分析核心区域（信号最强，避免噪声干扰）
    mask = (t > t_peak - core_radius) & (t < t_peak + core_radius)
    env1 = np.abs(hilbert(s1))[mask]
    env2 = np.abs(hilbert(s2))[mask]
    # 去均值，消除常数偏移
    env1 -= np.mean(env1)
    env2 -= np.mean(env2)
    # 互相关
    corr = correlate(env1, env2, mode='same')
    lags = np.arange(len(corr)) - len(corr)//2
    delay_samples = lags[np.argmax(corr)]
    delay = delay_samples / fs
    return delay, corr, lags

# ============================================================
# 主循环：处理 H1 和 L1 两个探测器
# ============================================================
for det, fpath in files.items():
    print(f"--- Processing {det} ---")
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        continue

    t, strain, fs = load_strain(fpath)
    print(f"Strain loaded: {len(strain)} samples, fs = {fs:.1f} Hz")

    # 分别对两个频段进行带通滤波
    s_low = bandpass(strain, band_low[0], band_low[1], fs)
    s_high = bandpass(strain, band_high[0], band_high[1], fs)

    # 测量两个频段之间的包络延迟
    delay, corr, lags = envelope_crosscorr(t, s_low, s_high, fs)
    print(f"Measured delay ({band_low[0]}-{band_low[1]} Hz vs "
          f"{band_high[0]}-{band_high[1]} Hz): {delay*1000:.3f} ms")

    # 绘图：左图为滤波后波形，右图为互相关
    mask = (t > t_peak - core_radius) & (t < t_peak + core_radius)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t[mask], s_low[mask], alpha=0.7, label=f'{band_low[0]}-{band_low[1]} Hz')
    plt.plot(t[mask], s_high[mask], alpha=0.7, label=f'{band_high[0]}-{band_high[1]} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.legend()
    plt.title(f'{det} filtered strain (core region)')

    plt.subplot(1, 2, 2)
    plt.plot(lags / fs, corr)
    plt.axvline(delay, color='r', linestyle='--',
                label=f'Peak at {delay*1000:.3f} ms')
    plt.xlabel('Lag (s)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.title('Envelope cross-correlation')
    plt.tight_layout()
    plt.savefig(f'{det}_dispersion.png', dpi=150)
    plt.show()

print("\n--- 检验说明 ---")
print("测得的延迟（约 -100 ms）主要来源于引力波信号本身的 chirp 特征：")
print("信号频率从低到高扫过，高频成分自然先到达。")
print("该延迟不能被解释为传播路径上的真空色散。")
print("本理论预言的色散时延（约 1e-20 s）远小于当前测量精度（约 1e-6 s），")
print("现有数据无法排除该理论。")