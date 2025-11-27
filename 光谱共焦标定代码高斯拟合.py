"""
光谱共焦 14000（实际用了第14面采集到的13854个点） 针孔阵列标定脚本（PNG 版）
量程 0-2.9 mm，30 张等间隔图片
"""
import os, glob
import numpy as np
import imageio.v2 as imageio
from skimage.feature import peak_local_max
from scipy.ndimage import center_of_mass
from numpy.polynomial import Polynomial

# ========== 1. 参数 ==========
os.chdir(r'C:\Users\10574\Desktop\针孔阵列图')   # 图片目录
files = sorted(glob.glob('*.png'))  # 生成排好序的文件名列表
N  = 13854                       # 针孔总数（第14面实测峰值数）
Δz = np.linspace(2.9, 0, 30)     # 30 个已知高度（mm）

# ========== 2. 读 30 张 PNG ==========
imgs = np.stack([imageio.imread(f).astype(np.float32) @ [0.3, 0.3, 0.4]
                 for f in files])   #读图后直接转化RGB → 灰度

ref = imgs[14]   # 以14帧作为参考帧

# 直接返回 (row, col) 即 (y, x)
coords = peak_local_max(ref,
                            min_distance=20,
                            threshold_abs=100)   # shape (N_peak, 2) 顺序 (y,x)


# ========== 4. 逐帧滑动 ROI（半径不变，中心随光斑移动） ==========

from scipy.optimize import curve_fit

def gauss_1d(x, A, mu, sigma, B):
    """一维高斯 + 常数偏移"""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B

def fit_gauss_center(line):
    """
    对一条 1-D 灰度分布做高斯拟合，返回亚像素峰值位置。
    失败时退而求其次用质心。
    """
    x = np.arange(len(line), dtype=float)
    try:
        # 初值：质心做 mu，sigma=2，A=max-min，B=min
        mu0 = np.sum(x * line) / np.sum(line)
        A0  = line.max() - line.min()
        B0  = line.min()
        p0  = [A0, mu0, 2.0, B0]
        popt, _ = curve_fit(gauss_1d, x, line, p0=p0, maxfev=1000)
        return popt[1]          # 高斯中心
    except RuntimeError:
        # 拟合失败，退回质心
        return center_of_mass(line)[0]

# ---------- 第 14 帧 ----------
peaks = np.empty((N, 30))
radius = 9
for k in range(N):
    y0, x0 = coords[k]
    y1, y2 = max(y0 - radius, 0), y0 + radius + 1
    x1, x2 = max(x0 - radius, 0), x0 + radius + 1
    roi = imgs[14, y1:y2, x1:x2]
    # 对 y 方向每一行求和，得到 1-D 信号
    proj = roi.sum(axis=1)
    dy = fit_gauss_center(proj)
    peaks[k, 14] = y1 + dy

# ---------- 第 13→0 帧 ----------
for i in range(13, -1, -1):
    for k in range(N):
        y_last = peaks[k, i + 1]
        x_last = coords[k, 1]
        y1 = max(int(y_last) - radius, 0)
        y2 = int(y_last) + radius + 1
        x1 = max(x_last - radius, 0)
        x2 = x_last + radius + 1
        roi = imgs[i, y1:y2, x1:x2]
        proj = roi.sum(axis=1)
        dy = fit_gauss_center(proj)
        peaks[k, i] = y1 + dy

# ---------- 第 15→29 帧 ----------
for i in range(15, 30):
    for k in range(N):
        y_last = peaks[k, i - 1]
        x_last = coords[k, 1]
        y1 = max(int(y_last) - radius, 0)
        y2 = int(y_last) + radius + 1
        x1 = max(x_last - radius, 0)
        x2 = x_last + radius + 1
        roi = imgs[i, y1:y2, x1:x2]
        proj = roi.sum(axis=1)
        dy = fit_gauss_center(proj)
        peaks[k, i] = y1 + dy

# ========== 5. 自动选阶并拟合 ==========
def fit_order(z, y, max_order=3, thresh=0.05):
    for deg in range(1, max_order + 1):
        p = Polynomial.fit(z, y, deg)
        res = np.std(y - p(z))
        if deg > 1 and (res_old - res) < thresh:
            return deg - 1
        res_old = res
    return max_order


coeffs = []                     # 仍用 list 存各条曲线
orders = np.empty(N, dtype=int) # 阶数保持 int
for k in range(N):
    deg = fit_order(Δz, peaks[k])
    p   = Polynomial.fit(Δz, peaks[k], deg)
    coeffs.append(p.convert().coef[::-1])
    orders[k] = deg

# ========== 6. 保存标定表 ==========
# 把 coeffs 转成 object 数组，避免“非同质”报错
coeffs_np = np.empty(len(coeffs), dtype=object)
coeffs_np[:] = coeffs           # 逐条塞进去

np.savez('calib_table_14k_png.npz',
         coeffs=coeffs_np,
         orders=orders,
         x0=coords[:, 1],
         y0=coords[:, 0])


print('标定完成！已生成 calib_table_14k_png.npz')
miss = np.isnan(peaks).sum(axis=1)   # 每孔丢失帧数
if miss.max() > 0:

    print('警告：有', (miss > 0).sum(), '个光斑在部分帧丢失！')
