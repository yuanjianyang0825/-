# -*- coding: utf-8 -*-
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
files = sorted(glob.glob('*.png'))
N  = 13854                       # 针孔总数（第14面实测峰值数）
Δz = np.linspace(2.9, 0, 30)     # 30 个已知高度（mm）

# ========== 2. 读 30 张 PNG ==========
imgs = np.stack([imageio.imread(f).astype(np.float32) @ [0.299, 0.587, 0.114]
                 for f in files])   #读图后直接转化RGB → 灰度

ref = imgs[14]

# 直接返回 (row, col) 即 (y, x)
coords = peak_local_max(ref,
                            min_distance=20,
                            threshold_abs=100)   # shape (N_peak, 2) 顺序 (y,x)


# ========== 4. 逐帧滑动 ROI（半径不变，中心随光斑移动） ==========
peaks = np.empty((N, 30))
radius = 9                      # 初始半径，可略大于原来
# 第 14 帧先算一次
for k in range(N):
    y0, x0 = coords[k]          # 第 0 帧中心
    y1, y2 = max(y0 - radius, 0), y0 + radius + 1
    roi = imgs[14, y1:y2, x0 - radius:x0 + radius + 1]
    dy = center_of_mass(roi)[0]
    peaks[k, 14] = y1 + dy       # 全局 y

# 第 13→0 帧：用上一帧结果做新中心
for i in range(13,-1,-1):
    for k in range(N):
        y_last = peaks[k, i+1]              # 上一帧的亚像素 y
        x_last = coords[k, 1]               # x 几乎不变，仍用初始 x0
        y1 = max(int(y_last) - radius, 0)
        y2 = int(y_last) + radius + 1
        roi = imgs[i, y1:y2, x_last - radius:x_last + radius + 1]
        dy = center_of_mass(roi)[0]
        peaks[k, i] = y1 + dy
for i in range(15, 30):
    for k in range(N):
        y_last = peaks[k, i-1]              # 上一帧的亚像素 y
        x_last = coords[k, 1]               # x 几乎不变，仍用初始 x0
        y1 = max(int(y_last) - radius, 0)
        y2 = int(y_last) + radius + 1
        roi = imgs[i, y1:y2, x_last - radius:x_last + radius + 1]
        dy = center_of_mass(roi)[0]
        peaks[k, i] = y1 + dy               # 更新全局 y

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
