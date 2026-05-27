"""
=============================================================================
模块名称: eval_sfmm.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统
作　　者: TonyNee
创建日期: 2026-01-13
最后修改: 2026-05-27
=============================================================================

功能概述:
  轻量级评估脚本, 在计算标准指标 (MAE/MSE/RMSE) 的基础上,
  增加了推理时间测量和测量值/预测值直方图对比可视化。
  适用于需要快速验证模型推理速度的场景。

包含功能:
  1. 加载配置和模型
  2. 计算推理时间 (总时间 / 单样本平均时间)
  3. 计算 MAE/MSE/RMSE 指标
  4. 保存预测 CSV
  5. 绘制测量值直方图 vs 预测值直方图 (双子图对比)

使用方式:
  python eval_sfmm.py --mdir ./output/resmlp/20251218120000/
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import time

from config import Config


# =============================================================================
# 阶段 0: 配置环境
# =============================================================================
parser = argparse.ArgumentParser(description="THz 频率校准模型轻量评估 + 直方图")
parser.add_argument("--mdir", required=True,
                    help="模型输出目录路径")
args = parser.parse_args()

Config.load_yaml(args.mdir)
device = torch.device(Config.DEVICE)
print(f"Device: {device}")

# =============================================================================
# 阶段 1: 加载评估数据
# =============================================================================
df = pd.read_csv(Config.EVAL_CSV)
x_eval = df.iloc[:, 0].values.astype(np.float32)                # 测量频率 (GHz)
y_eval = df.iloc[:, 1].values.astype(np.float32)                # 真实频率 (GHz)

X = torch.tensor(x_eval.reshape(-1, 1)).to(device)
Y = torch.tensor(y_eval.reshape(-1, 1)).to(device)

# =============================================================================
# 阶段 2: 加载模型
# =============================================================================
model = Config.MODEL_CLASS().to(device)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
model.eval()

# =============================================================================
# 阶段 3: 推理 & 计时
# =============================================================================
# 使用 perf_counter 进行高精度计时
start_time = time.perf_counter()
with torch.no_grad():
    y_pred = model(X).cpu().numpy()
end_time = time.perf_counter()

# 推理时间统计
total_time_ms = (end_time - start_time) * 1000                   # 毫秒
avg_time_us = (end_time - start_time) / len(X) * 1e6             # 微秒/样本

print("\n===== 推理时间 =====")
print(f"总样本数: {len(X)}")
print(f"总推理时间: {total_time_ms:.3f} ms")
print(f"平均每样本: {avg_time_us:.3f} μs ({avg_time_us/1000:.5f} ms)")

y_true = Y.cpu().numpy()
x_true = X.cpu().numpy()

# =============================================================================
# 阶段 4: 误差指标计算
# =============================================================================
meas_residuals = (x_true - y_true).flatten()                    # 原始测量误差
pred_residuals = (y_pred - y_true).flatten()                    # 模型修正后残差

MAE = float(np.mean(np.abs(pred_residuals)) * 1000)              # MHz
MSE = float(np.mean(pred_residuals ** 2))                        # GHz^2
RMSE = float(np.sqrt(MSE) * 1000)                                # MHz

metrics = {"MAE": MAE, "MSE": MSE, "RMSE": RMSE}
Config.update_yaml(model_dir=args.mdir, metrics_dict=metrics)

print("\n===== ResMLP 评估结果 =====")
print(f"MAE: {MAE:.10f} MHz")
print(f"MSE: {MSE:.10f} GHz^2")
print(f"RMSE: {RMSE:.10f} MHz\n")

# =============================================================================
# 阶段 5: 保存预测结果 CSV
# =============================================================================
model_name = Config.MODEL_TYPE.replace(" ", "_")

df_out = pd.DataFrame({
    "F_test_GHz": x_eval,
    "F_true_GHz": y_eval,
    "F_pred_GHz": y_pred.flatten(),
    "Residual(MHz)": pred_residuals * 1000                       # 转换为 MHz
})
df_out.to_csv(os.path.join(Config.RESULT_SAVE_DIR, f"data_predicted_{model_name}.csv"), index=False)

# =============================================================================
# 阶段 6: 绘制测量值与预测值直方图对比
# =============================================================================
true_value = 200                                                 # 参考真值, 用于对比标记

# 创建 1 行 2 列子图布局
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'{Config.MODEL_TYPE} Model Evaluation Visualization', fontsize=16)

# ---- 左图: 测量值分布直方图 ----
axes[0].hist(x_eval, bins=20, facecolor='blue', edgecolor='black',
             alpha=0.7, label='Measured Values')
axes[0].axvline(x=true_value, color='red', linestyle='--', linewidth=2,
                label=f'True Value = {true_value} GHz')          # 红色虚线标注真值
axes[0].axvline(x=np.mean(x_eval), color='green', linestyle='-', linewidth=2,
                label=f'Meas Mean = {np.mean(x_eval):.6f} GHz')  # 绿色实线标注均值
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Frequency (GHz)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Measured Values Distribution Histogram', fontsize=14)
axes[0].legend(loc='best', fontsize=10)

# ---- 右图: 预测值分布直方图 ----
axes[1].hist(y_pred.flatten(), bins=20, facecolor='orange', edgecolor='black',
             alpha=0.7, label='Predicted Values')
axes[1].axvline(x=true_value, color='red', linestyle='--', linewidth=2,
                label=f'True Value = {true_value} GHz')
axes[1].axvline(x=np.mean(y_pred.flatten()), color='green', linestyle='-', linewidth=2,
                label=f'Pred Mean = {np.mean(y_pred.flatten()):.6f} GHz')
axes[1].set_xlabel('Frequency (GHz)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Predicted Values Distribution Histogram', fontsize=14)
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, f'evaluation_visualization_{model_name}.png'),
            dpi=300, bbox_inches='tight')
plt.show()
