# -*- coding: utf-8 -*-
"""
基于 calib_table_14k_png.npz 的三维重建脚本
采用「先按 X 匹配，再在候选中选 Y 最近」的策略
适用于：Y方向位移大、X基本不变的针孔阵列系统
"""
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage.feature import peak_local_max
from scipy.ndimage import center_of_mass
from numpy.polynomial import Polynomial
import os

# ========== 1. 加载标定数据 ==========
os.chdir(r'C:\Users\10574\Desktop\针孔阵列图')  # 确保路径正确

# 必须加 allow_pickle=True
calib = np.load('calib_table_14k_png.npz', allow_pickle=True)
coeffs = calib['coeffs']        # 多项式系数列表 (13854,)
orders = calib['orders']        # 阶数
x0_calib = calib['x0']          # 第14帧初始 x 坐标
y0_calib = calib['y0']          # 第14帧初始 y 坐标
N = len(x0_calib)               # 应该是 13854

print(f"✅ 已加载标定数据：共 {N} 个有效针孔（基于第14帧）")

# ========== 2. 读取新样品图像 ==========
img_path = 'sample.png'  # ← 修改为你自己的图像名
if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ 找不到图像文件: {img_path}")

img = imageio.imread(img_path)

# 转灰度
if img.ndim == 3:
    gray = img @ [0.299, 0.587, 0.114]
else:
    gray = img.copy()
gray = gray.astype(np.float32)

print(f"✅ 已加载待测图像: {img_path}, 尺寸 {gray.shape}")

# ========== 3. 检测光斑粗略位置 ==========
peaks_new = peak_local_max(
    gray,
    min_distance=20,
    threshold_abs=100
)  # 输出 (y, x)

print(f"🔍 检测到 {len(peaks_new)} 个光斑候选点")
if len(peaks_new) == 0:
    raise ValueError("⚠️ 未检测到任何光斑，请检查图像质量或调整 threshold_abs")

# ========== 4. 亚像素精确定位（Y方向）==========
radius = 9
y_subpix = np.zeros(len(peaks_new))

for i, (y_int, x_int) in enumerate(peaks_new):
    y1 = max(int(y_int) - radius, 0)
    y2 = int(y_int) + radius + 1
    x1 = max(int(x_int) - radius, 0)
    x2 = int(x_int) + radius + 1
    roi = gray[y1:y2, x1:x2]

    if roi.size == 0 or roi.sum() == 0:
        y_subpix[i] = np.nan
    else:
        try:
            dy, dx = center_of_mass(roi)
            y_subpix[i] = y1 + dy
        except:
            y_subpix[i] = np.nan

# 当前图像中的光斑位置
new_x = peaks_new[:, 1]        # detected x (col)
new_y = peaks_new[:, 0]        # detected y (row)，用于显示
y_measured = y_subpix          # 亚像素 y 坐标

# ========== 5. 两级匹配：先 X 近 → 再 Y 最近 ==========
matches = -np.ones(len(new_x), dtype=int)  # -1 表示未匹配

# 设置 X 匹配容忍范围（单位：像素）
# 根据实际光斑间距设置，比如间隔 ~30px，则 ±15 合理
x_tolerance = 15

for i in range(len(new_x)):
    xi = new_x[i]
    yi = y_measured[i]
    if np.isnan(yi):
        continue

    # Step 1: 找出所有 x 在 [xi - tol, xi + tol] 范围内的标定点
    candidate_mask = (x0_calib >= xi - x_tolerance) & (x0_calib <= xi + x_tolerance)
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        continue  # 无候选点

    # Step 2: 在这些候选点中，找 y0 最接近当前 yi 的那个
    y0_candidates = y0_calib[candidate_indices]
    dy = np.abs(y0_candidates - yi)
    best_idx_in_candidates = np.argmin(dy)
    matched_k = candidate_indices[best_idx_in_candidates]

    matches[i] = matched_k

num_matched = np.sum(matches >= 0)
print(f"✅ 成功匹配 {num_matched} 个光斑到标定库")

# ========== 6. 解算高度 Z ==========
Z_heights = np.full(len(matches), np.nan)
z_search = np.linspace(0.0, 2.9, 2000)  # 高分辨率搜索

for i, match_idx in enumerate(matches):
    if match_idx < 0 or np.isnan(y_measured[i]):
        continue

    coef = coeffs[match_idx]
    p = Polynomial(coef[::-1])  # 升幂排列
    y_pred = p(z_search)
    best_z_idx = np.argmin((y_pred - y_measured[i]) ** 2)
    Z_heights[i] = z_search[best_z_idx]

# ========== 7. 提取有效数据点 ==========
valid_mask = ~np.isnan(Z_heights)
x_coords = new_x[valid_mask].astype(int)       # 图像 x 像素
y_coords = new_y[valid_mask].astype(int)       # 图像 y 像素（整数，用于显示）
z_values = Z_heights[valid_mask]               # 高度值 (mm)

if len(z_values) == 0:
    raise ValueError("❌ 未能解算出任何有效高度！请检查图像对焦或标定文件。")

print(f"📊 有效高度点数: {len(z_values)} / {len(Z_heights)}")

# ========== 8. 插值生成连续高度图 ==========
xi_grid = np.linspace(x_coords.min(), x_coords.max(), 500)
yi_grid = np.linspace(y_coords.min(), y_coords.max(), 500)
Xi, Yi = np.meshgrid(xi_grid, yi_grid)

Zi = griddata(
    points=(x_coords, y_coords),
    values=z_values,
    xi=(Xi, Yi),
    method='cubic',
    fill_value=np.nan
)

# ========== 9. 可视化结果 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 10))

# 子图1：原始图像 + 检测点
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.plot(x_coords, y_coords, 'r.', markersize=2, alpha=0.8)
plt.title("检测到的光斑位置")
plt.axis('off')

# 子图2：散点高度分布
plt.subplot(2, 2, 2)
sc = plt.scatter(x_coords, y_coords, c=z_values, cmap='Spectral_r', s=10, vmin=0, vmax=2.9)
plt.colorbar(sc, label='高度 Z (mm)')
plt.title("各光斑高度分布")
plt.axis('off')

# 子图3：插值后的三维形貌
plt.subplot(2, 1, 2)
im = plt.imshow(
    Zi,
    cmap='Spectral_r',
    extent=[xi_grid.min(), xi_grid.max(), yi_grid.min(), yi_grid.max()],
    origin='lower',
    vmin=0,
    vmax=2.9
)
plt.colorbar(im, label='高度 Z (mm)')
plt.title("重建的三维表面形貌（插值后）")
plt.xlabel("图像 X 像素")
plt.ylabel("图像 Y 像素")

plt.tight_layout()
plt.show()

# ========== 10. 保存结果 ==========
np.savez(
    'reconstructed_3d_surface_Xfirst.npz',
    x=x_coords,
    y=y_coords,
    z=z_values,
    Xi=Xi,
    Yi=Yi,
    Zi=Zi,
    params={
        'range_mm': (0.0, 2.9),
        'N_frames': 30,
        'ref_frame': 14,
        'N_spots': N,
        'x_tolerance': x_tolerance
    }
)

print("🎉 三维形貌重建完成！")
print(f"   有效点数: {len(z_values)}")
print(f"   高度范围: {z_values.min():.3f} ~ {z_values.max():.3f} mm")
print("   数据已保存至: reconstructed_3d_surface_Xfirst.npz")