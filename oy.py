import os, glob
import numpy as np
import imageio.v2 as imageio
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass

# ========== 路径 ==========
dark_dir  = r"C:\Users\A\Desktop\20ims"
light_dir = r"C:\Users\A\Desktop\100ims"

# ========== 安全转灰度函数 ==========
def to_gray(img):
    img = img.astype(np.float32)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]  # 去掉 Alpha 通道
        elif img.shape[2] != 3:
            raise ValueError(f"Unexpected channel number: {img.shape[2]}")
        return img @ [0.3, 0.3, 0.4]
    elif img.ndim == 2:
        return img
    else:
        raise ValueError("Invalid image dimension")

# ========== 读暗场 ==========
print("正在加载暗场图像...")
os.chdir(dark_dir)
dark_files = sorted(glob.glob('*.bmp'))
assert len(dark_files) >= 20
dark_imgs = [to_gray(imageio.imread(f)) for f in dark_files[:20]]
bg_mean = np.mean(np.stack(dark_imgs), axis=0)

# ========== 读亮场 ==========
print("正在加载亮场图像...")
os.chdir(light_dir)
light_files = sorted(glob.glob('*.bmp'))
assert len(light_files) >= 50
light_imgs = [to_gray(imageio.imread(f)) for f in light_files[:50]]
imgs = np.clip(np.stack(light_imgs) - bg_mean, 0, None)

# ========== 后续分析（不变）==========
ref_img = imgs[0]
coords = peak_local_max(ref_img, min_distance=20, threshold_abs=50)
N = coords.shape[0]
print(f"检测到亮点数: {N}")

# ...（其余代码完全不变，从高斯拟合开始复制即可）
# ========== 5. 高斯拟合函数 ==========
def gauss_1d(x, A, mu, sigma, B):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B

def fit_gauss_center(line):
    x = np.arange(len(line), dtype=float)
    try:
        mu0 = np.sum(x * line) / np.sum(line)
        A0 = line.max() - line.min()
        B0 = line.min()
        p0 = [A0, mu0, 2.0, B0]
        popt, _ = curve_fit(gauss_1d, x, line, p0=p0, maxfev=1000)
        return popt[1]
    except (RuntimeError, ValueError):
        return center_of_mass(line)[0]

# ========== 6. 逐帧拟合（固定参考坐标）==========
print("正在逐帧拟合光斑位置...")
radius = 9
all_peaks_y = np.full((N, 50), np.nan, dtype=np.float32)

for frame_idx in range(50):
    img = imgs[frame_idx]
    for k in range(N):
        y0_ref, x0_ref = coords[k]
        y1 = max(int(y0_ref) - radius, 0)
        y2 = int(y0_ref) + radius + 1
        x1 = max(int(x0_ref) - radius, 0)
        x2 = int(x0_ref) + radius + 1
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0 or roi.sum() < 1e-3:
            continue
        proj = roi.sum(axis=1)
        dy = fit_gauss_center(proj)
        all_peaks_y[k, frame_idx] = y1 + dy

# ========== 7. 漂移分析 ==========
print("\n" + "="*50)
print("📊 漂移分析结果")
print("="*50)

frame_mean_y = np.nanmean(all_peaks_y, axis=0)
frame_std_over_time = np.std(frame_mean_y)
print(f"• 每帧平均y位置的标准差: {frame_std_over_time:.5f} 像素")

spot_std = np.nanstd(all_peaks_y, axis=1)
valid_spots = ~np.isnan(spot_std)
spot_std_valid = spot_std[valid_spots]

print(f"• 单个光斑y位置稳定性:")
print(f"    中位数 std: {np.median(spot_std_valid):.5f} 像素")
print(f"    平均 std:   {np.mean(spot_std_valid):.5f} 像素")
print(f"    最大 std:   {np.max(spot_std_valid):.5f} 像素")
print(f"    >0.05像素的光斑数: {np.sum(spot_std_valid > 0.05)} / {valid_spots.sum()}")

# 趋势分析
from scipy.stats import linregress
slope, _, _, p_val, _ = linregress(np.arange(50), frame_mean_y)
trend_str = "显著" if p_val < 0.05 else "不显著"
print(f"• 整体线性漂移趋势: p = {p_val:.3f} → {trend_str}")

print("\n✅ 分析完成！")