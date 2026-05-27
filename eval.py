"""
=============================================================================
模块名称: eval.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统
作　　者: TonyNee
创建日期: 2025-12-20
最后修改: 2026-05-27
=============================================================================

功能概述:
  本模块实现训练后模型的全面评估流程, 包括:
  1. 加载 config.yaml 恢复训练配置
  2. 加载模型权重, 在评估集上推理
  3. 计算多维误差指标 (MAE/MSE/RMSE/R2 + 分位数误差 E1s/E2s/E3s)
  4. 保存预测结果 CSV
  5. 绘制测量误差 vs 预测误差的残差对比图 (含分频段最大误差标注)
  6. 绘制预测残差的概率密度分布 (PDF) 直方图, 叠加高斯拟合曲线
  7. 计算分通道 (低频/高频) MAE

使用方式:
  python eval.py --mdir ./output/resmlp/20251218120000/
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

from config import Config, Utils
from scipy.stats import gaussian_kde


# =============================================================================
# 阶段 0: 配置环境
# =============================================================================
parser = argparse.ArgumentParser(description="THz 频率校准模型评估脚本")
parser.add_argument("--mdir", required=True,
                    help="模型输出目录路径, 内含 config.yaml 和 .pth 权重文件")
args = parser.parse_args()

# 从训练时保存的 config.yaml 恢复全部配置
Config.load_yaml(args.mdir)
device = torch.device(Config.DEVICE)
print(f"Device: {device}")

# =============================================================================
# 阶段 1: 加载评估数据
# =============================================================================
df = pd.read_csv(Config.EVAL_CSV)
x_eval = df.iloc[:, 0].values.astype(np.float32)                # 测量频率 (GHz)
y_eval = df.iloc[:, 1].values.astype(np.float32)                # 真实频率 (GHz)

# 转为 GPU 张量
X = torch.tensor(x_eval.reshape(-1, 1)).to(device)
Y = torch.tensor(y_eval.reshape(-1, 1)).to(device)

# =============================================================================
# 阶段 2: 加载训练好的模型
# =============================================================================
model = Config.MODEL_CLASS().to(device)
# 从标准路径加载权重, map_location 确保 CPU/CUDA 兼容
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
model.eval()                                                    # 切换到推理模式

# =============================================================================
# 阶段 3: 模型推理 (预测)
# =============================================================================
with torch.no_grad():                                           # 禁止梯度计算
    y_pred = model(X).cpu().numpy()

# 将张量移回 CPU 转为 numpy 数组
y_true = Y.cpu().numpy()
x_true = X.cpu().numpy()

# =============================================================================
# 阶段 4: 误差指标计算
# =============================================================================
# 残差定义:
#   meas_residuals = 测量值 - 真实值  (原始测量偏差)
#   pred_residuals = 预测值 - 真实值  (模型修正后的残差)
meas_residuals = (x_true - y_true).flatten()
pred_residuals = (y_pred - y_true).flatten()

# 测量误差指标 (模型修正前的基线)
MAE_m = float(np.mean(np.abs(meas_residuals)) * 1000)           # MHz
MSE_m = float(np.mean(meas_residuals ** 2))                     # GHz^2
RMSE_m = float(np.sqrt(MSE_m) * 1000)                           # MHz

print("\n===== EVAL 初始指标 =====")
print(f"MAE: {MAE_m:.2f} MHz")
print(f"RMSE: {RMSE_m:.2f} MHz")

# ---- 分位数误差 (单次测量可靠性指标) ----
abs_err_mhz = np.abs(pred_residuals) * 1000                     # 预测残差绝对值 (MHz)

# 1σ/2σ/3σ 分位数: 68.25%/95.45%/99.73% 置信度的误差上限
E1sigma = float(np.percentile(abs_err_mhz, 68.25))
E2sigma = float(np.percentile(abs_err_mhz, 95.45))
E3sigma = float(np.percentile(abs_err_mhz, 99.73))

# 常用分位数: 80%/90%/95%/99% 误差上限
E80 = float(np.percentile(abs_err_mhz, 80))
E90 = float(np.percentile(abs_err_mhz, 90))
E95 = float(np.percentile(abs_err_mhz, 95))
E99 = float(np.percentile(abs_err_mhz, 99))

# 标准回归指标
MAE = float(np.mean(np.abs(pred_residuals)) * 1000)             # MHz
MSE = float(np.mean(pred_residuals ** 2))                       # GHz^2
RMSE = float(np.sqrt(MSE) * 1000)                               # MHz

# R² 决定系数: 1 - SS_res/SS_tot
SS_res = np.sum(pred_residuals ** 2)                            # 残差平方和
SS_tot = np.sum((y_true - np.mean(y_true)) ** 2)                # 总平方和
R2 = float(1 - SS_res / SS_tot)

# 组装评估指标字典
metrics = {
    "MAE_MHz": MAE,
    "RMSE_MHz": RMSE,
    "MSE_GHz2": MSE,
    "R2": R2,
    "E1s_MHz": E1sigma,
    "E2s_MHz": E2sigma,
    "E3s_MHz": E3sigma,
    "E80_MHz": E80,
    "E90_MHz": E90,
    "E95_MHz": E95,
    "E99_MHz": E99
}

# 将指标写回 config.yaml (持久化评估结果)
Config.update_yaml(model_dir=args.mdir, metrics_dict=metrics)

# ---- 打印评估结果 ----
print("\n===== ResMLP 评估结果 =====")
print(f"MAE: {MAE:.10f} MHz")
print(f"MSE: {MSE:.10f} GHz^2")
print(f"RMSE: {RMSE:.10f} MHz")
print(f"R2: {R2:.10f}")
print(f"E1sigma: {E1sigma:.4f} MHz")
print(f"E2sigma: {E2sigma:.4f} MHz")
print(f"E3sigma: {E3sigma:.4f} MHz")
print(f"E80: {E80:.4f} MHz")
print(f"E90: {E90:.4f} MHz")
print(f"E95: {E95:.4f} MHz")
print(f"E99: {E99:.4f} MHz")

# =============================================================================
# 阶段 5: 保存预测结果 CSV
# =============================================================================
model_name = Config.MODEL_TYPE.replace(" ", "_")

df_out = pd.DataFrame({
    "F_test_GHz": x_eval,                                       # 测试频率 (GHz)
    "F_true_GHz": y_eval,                                       # 真实频率 (GHz)
    "F_pred_GHz": y_pred.flatten(),                             # 模型预测频率 (GHz)
    "Residual(MHz)": pred_residuals * 1000                      # 预测残差 (MHz)
})
df_out.to_csv(os.path.join(Config.RESULT_SAVE_DIR, f"data_predicted_{model_name}.csv"), index=False)

# =============================================================================
# 阶段 6: 绘制残差对比图
# =============================================================================
x_freq = y_eval                                                 # 以真实频率为 X 轴
idx = np.argsort(x_freq)                                        # 按频率升序排列

save_name = f"residual_plot_{model_name}.png"
title_str = f"{Config.MODEL_TYPE} Frequency Residuals"

plt.figure(figsize=(10, 6))

# 绘制测量误差和模型修正后误差
plt.plot(x_freq[idx], meas_residuals[idx] * 1000, label='MEASURED ERROR')
plt.plot(x_freq[idx], pred_residuals[idx] * 1000, label='PREDICTED ERROR')

# 零残差参考线
plt.axhline(0, linestyle="--", color="black")

# 747 GHz 通道分界线
plt.axvline(747, linestyle=':', color='gray', alpha=0.7)
y_min, y_max = plt.ylim()
plt.text(747, y_min - (y_max - y_min) * 0.018, '747',
         ha='center', va='top', fontsize=10, color='gray', alpha=0.8)

plt.xlabel("Frequency (GHz)")
plt.ylabel("Residuals (MHz)")
plt.title(title_str)
plt.grid(True)

# 计算整体 MAE
meas_mae = np.mean(np.abs(meas_residuals)) * 1000
pred_mae = np.mean(np.abs(pred_residuals)) * 1000

# 使用 Utils 标注各频段最大绝对误差位置
meas_results = Utils.annotate_max_abs_by_range(x_freq[idx], meas_residuals[idx] * 1000, 'MEAS', 0)
pred_results = Utils.annotate_max_abs_by_range(x_freq[idx], pred_residuals[idx] * 1000, 'PRED', 1)

# 构建左上角信息表 (MEAS vs PRED 对比)
info_text = "   Metric              MEAS         PRED\n"
info_text += "-" * 50 + "\n"
meas_max_all = np.max(np.abs(meas_residuals[idx] * 1000))
pred_max_all = np.max(np.abs(pred_residuals[idx] * 1000))
info_text += f"MAE (MHz)        {meas_mae:>7.2f}    {pred_mae:>7.2f}\n"
info_text += f"MAX (MHz)        {meas_max_all:>7.2f}    {pred_max_all:>7.2f}"

plt.text(0.015, 0.30, info_text,
         transform=plt.gca().transAxes,                         # 轴坐标系统
         verticalalignment='top',
         fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, save_name))
plt.close()


# =============================================================================
# 阶段 7: 分通道 (低频/高频) MAE 计算
# =============================================================================
# 747 GHz 是 THz 系统双通道的分界频率
threshold = 747

# ---- 低频段 (<747 GHz) ----
low_freq_mask = x_freq < threshold
high_freq_mask = x_freq > threshold

if np.any(low_freq_mask):
    low_freq_meas_residuals = meas_residuals[low_freq_mask]
    low_freq_pred_residuals = pred_residuals[low_freq_mask]

    low_freq_meas_mae = np.mean(np.abs(low_freq_meas_residuals)) * 1000   # MHz
    low_freq_pred_mae = np.mean(np.abs(low_freq_pred_residuals)) * 1000   # MHz
    low_freq_count = np.sum(low_freq_mask)
else:
    low_freq_meas_mae = 0
    low_freq_pred_mae = 0
    low_freq_count = 0

# ---- 高频段 (>747 GHz) ----
if np.any(high_freq_mask):
    high_freq_meas_residuals = meas_residuals[high_freq_mask]
    high_freq_pred_residuals = pred_residuals[high_freq_mask]

    high_freq_meas_mae = np.mean(np.abs(high_freq_meas_residuals)) * 1000   # MHz
    high_freq_pred_mae = np.mean(np.abs(high_freq_pred_residuals)) * 1000   # MHz
    high_freq_count = np.sum(high_freq_mask)
else:
    high_freq_meas_mae = 0
    high_freq_pred_mae = 0
    high_freq_count = 0

# 打印分通道结果
print("\n===== 分通道评估结果 =====")
print(f"CH1: 低频段 (Freq < {threshold} GHz, {low_freq_count} points):")
print(f"  MEAS_MAE: {low_freq_meas_mae:.4f} MHz")
print(f"  PRED_MAE: {low_freq_pred_mae:.4f} MHz")

print(f"\nCH2: 高频段 (Freq > {threshold} GHz, {high_freq_count} points):")
print(f"  MEAS_MAE: {high_freq_meas_mae:.4f} MHz")
print(f"  PRED_MAE: {high_freq_pred_mae:.4f} MHz")


# =============================================================================
# 阶段 8: 预测残差概率分布 (PDF) 与高斯拟合
# =============================================================================
# 将残差从 GHz 转换为 MHz 显示
residual_mhz = pred_residuals * 1000

# 统计量
mu = np.mean(residual_mhz)                                      # 均值 (理想为 0)
sigma = np.std(residual_mhz, ddof=1)                            # 无偏标准差

# X 轴范围取 0.5-99.5 百分位 (去除极端离群值影响)
x_min, x_max = np.percentile(residual_mhz, [0.5, 99.5])
x_pdf = np.linspace(x_min, x_max, 1000)

# 计算理论高斯 PDF
gaussian_pdf = (
    1 / (np.sqrt(2 * np.pi) * sigma)
    * np.exp(-0.5 * ((x_pdf - mu) / sigma) ** 2)
)

# ---- 绘图 ----
plt.figure(figsize=(8, 5))

# 直方图 (密度模式, 80 bins)
plt.hist(
    residual_mhz,
    bins=80,
    density=True,
    alpha=0.6,
    label="Prediction Residuals (PDF)"
)

# 叠加高斯拟合曲线
plt.plot(
    x_pdf,
    gaussian_pdf,
    'r-',
    linewidth=2,
    label=f'Gaussian Fit ($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})'
)

# 标注 1σ/2σ/3σ 分界线
for k in [1, 2, 3]:
    plt.axvline(mu + k * sigma, color='k', linestyle='--', alpha=0.6)
    plt.axvline(mu - k * sigma, color='k', linestyle='--', alpha=0.6)

# 3σ 文本说明
plt.text(
    mu + 3 * sigma,
    plt.ylim()[1] * 0.85,
    r'$3\sigma$',
    ha='right',
    va='top',
    fontsize=10
)

plt.xlabel("Prediction Residual (MHz)")
plt.ylabel("Probability Density")
plt.title(f"{Config.MODEL_TYPE} Residual Distribution")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存高清图片
pdf_save_name = os.path.join(
    Config.RESULT_SAVE_DIR,
    f"residual_pdf_{model_name}.png"
)
plt.savefig(pdf_save_name, dpi=300)
plt.close()

print(f"\n残差概率分布图已保存至: {pdf_save_name}")
