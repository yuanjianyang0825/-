import os, glob
import numpy as np
import imageio.v2 as imageio
from skimage.feature import peak_local_max
from scipy.ndimage import center_of_mass

# ========== 路径 ==========
dark_dir  = r"C:\Users\A\Desktop\20ims"
light_dir = r"C:\Users\A\Desktop\100ims"

# ========== 安全转灰度函数 ==========
def to_gray(img):
    img = img.astype(np.float32)
    if img.ndim == 3:
        return img[:, :, 0]  # 直接取 R 通道（等于 G=B）
    elif img.ndim == 2:
        return img
    else:
        raise ValueError("无效图像维度")

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

# ========== 检测参考亮点 ==========
ref_img = imgs[0]
coords = peak_local_max(ref_img, min_distance=20, threshold_abs=50)
N = coords.shape[0]
print(f"检测到亮点数: {N}")

# ========== 6. 逐帧拟合（使用质心法）==========
print("正在逐帧拟合光斑位置（质心法）...")
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
        
        # 沿 x 方向积分，得到 y 方向 1D 信号
        proj = roi.sum(axis=1)
        
        # 使用质心法计算亚像素 y 坐标
        try:
            dy = center_of_mass(proj)[0]
            all_peaks_y[k, frame_idx] = y1 + dy
        except:
            # 极少数情况（如全零）跳过
            continue

# ========== 7. 漂移分析 ==========
print("\n" + "="*50)
print("📊 漂移分析结果（质心法）")
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

print("\n✅ 分析完成！（使用质心法）")