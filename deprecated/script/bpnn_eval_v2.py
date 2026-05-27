"""
=============================================================================
模块名称: deprecated/script/bpnn_eval_v2.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (v2 BPNN min-max 评估, 已废弃)
作　　者: TonyNee
创建日期: 2025-11
最后修改: 2026-05-27
=============================================================================

功能概述:
  v2 BPNN 模型评估脚本, 配合 bpnn_train_v2.py。
  加载 Min-Max 归一化参数和训练好的论文 BPNN 模型,
  对测试数据进行归一化 -> 推理 -> 反归一化 -> 指标计算。

评估内容:
  - 预测 vs 真实值对比图 (MHz)
  - MAE/MSE/R² 指标 (MHz)
  - 残差曲线图 (MHz)
  - 残差直方图 (误差分布)

废弃原因: 被 v3 (z-score) 和 root/eval.py 取代。

注意: 此文件已废弃。
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from model_paper import BPNN                                          # 论文 BPNN (500-50-10)

# =============================================================================
# 1. 设备配置
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# =============================================================================
# 2. 加载测试数据 + 归一化参数
# =============================================================================
test_csv_path = './data/eval_20251119.csv'
df_test = pd.read_csv(test_csv_path)

# 加载训练时保存的 Min-Max 归一化参数
norm_params_path = './model/normalization_params.npy'
norm_params = np.load(norm_params_path, allow_pickle=True).item()

x_test_raw = df_test.iloc[:, 0].values.astype(np.float32)         # 测量频率 (MHz)
y_test_raw = df_test.iloc[:, 1].values.astype(np.float32)         # 真实频率 (MHz)

# 使用训练时的参数进行归一化
x_test_norm = (x_test_raw - norm_params['x_min']) / (norm_params['x_max'] - norm_params['x_min'])
y_test_norm = (y_test_raw - norm_params['y_min']) / (norm_params['y_max'] - norm_params['y_min'])

X_test = torch.tensor(x_test_norm.reshape(-1, 1)).to(device)
Y_test = torch.tensor(y_test_norm.reshape(-1, 1)).to(device)

print(f'测试数据归一化完成')
print(f'输入范围: {x_test_raw.min():.2f} - {x_test_raw.max():.2f} MHz')
print(f'输出范围: {y_test_raw.min():.2f} - {y_test_raw.max():.2f} MHz')

# =============================================================================
# 3. 加载模型
# =============================================================================
model = BPNN().to(device)
model_path = './model/bpnn_model_20250928.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f'成功加载模型: {model_path}')

# =============================================================================
# 4. 预测 + 反归一化
# =============================================================================
with torch.no_grad():
    y_pred_norm = model(X_test)

# 反归一化: [0, 1] -> MHz
y_pred = y_pred_norm.cpu().numpy() * (norm_params['y_max'] - norm_params['y_min']) + norm_params['y_min']
y_true = y_test_raw                                               # 使用原始标准值

# =============================================================================
# 5. 绘制预测对比图 (MHz)
# =============================================================================
plt.figure(figsize=(8, 5))
plt.plot(y_true, label='Standard (True)', marker='o', markersize=3)
plt.plot(y_pred, label='Predicted', linestyle='--', marker='s', markersize=3)
plt.xlabel('Sample Index')
plt.ylabel('Frequency (MHz)')
plt.title('Test Set: True vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_dir = './result/'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'test_prediction_vs_truth.png'))
plt.close()

# =============================================================================
# 6. 评价指标 (在原始 MHz 单位)
# =============================================================================
y_pred_tensor = torch.tensor(y_pred.reshape(-1, 1)).to(device)
y_true_tensor = torch.tensor(y_true.reshape(-1, 1)).to(device)

criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()

mse = criterion_mse(y_pred_tensor, y_true_tensor).item()          # MHz²
mae = criterion_mae(y_pred_tensor, y_true_tensor).item()          # MHz
r2 = 1 - (torch.sum((y_true_tensor - y_pred_tensor) ** 2)
          / torch.sum((y_true_tensor - torch.mean(y_true_tensor)) ** 2)).item()

print(f'\n测试集评价指标 (原始MHz单位):')
print(f'测试集 MAE: {mae:.6f} MHz')
print(f'测试集 MSE: {mse:.6f} MHz²')
print(f'测试集 R²: {r2:.6f}')

# =============================================================================
# 7. 保存预测结果 + 误差统计
# =============================================================================
save_pred_path = os.path.join(output_dir, 'test_predictions.csv')
df_pred = pd.DataFrame({
    'Fexperiment_MHz': x_test_raw,
    'Fstandard_MHz': y_true,
    'Fpredicted_MHz': y_pred.flatten(),
    'Absolute_Error_MHz': (y_true - y_pred.flatten()),           # 绝对误差
    'Relative_Error_Percent': (y_true - y_pred.flatten()) / y_true * 100  # 相对误差(%)
})
df_pred.to_csv(save_pred_path, index=False)
print(f'预测结果已保存至 {save_pred_path}')

# =============================================================================
# 8. 残差曲线图 (MHz)
# =============================================================================
residuals = (y_true - y_pred.flatten())                           # 残差 (MHz)

print(f"\n残差统计:")
print(f"残差最大值: {np.max(residuals):.6f} MHz")
print(f"残差最小值: {np.min(residuals):.6f} MHz")
print(f"残差绝对值最大值: {np.max(np.abs(residuals)):.6f} MHz")
print(f"残差标准差: {np.std(residuals):.6f} MHz")

plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals', color='orange', marker='o', markersize=4)
plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero Error')
plt.axhline(np.mean(residuals), color='red', linestyle='-', linewidth=1,
            label=f'Mean: {np.mean(residuals):.3f} MHz')         # 红色均值线
plt.xlabel('Sample Index')
plt.ylabel('Residual (MHz)')
plt.title('Residual Plot (Standard - Predicted)')
plt.legend()
plt.grid(True)
plt.tight_layout()

residual_plot_path = os.path.join(output_dir, 'residual_plot.png')
plt.savefig(residual_plot_path)
plt.close()
print(f'残差图已保存至 {residual_plot_path}')

# =============================================================================
# 9. 误差分布直方图
# =============================================================================
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(residuals), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(residuals):.3f} MHz')
plt.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero Error')
plt.xlabel('Residual (MHz)')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

histogram_path = os.path.join(output_dir, 'residual_histogram.png')
plt.savefig(histogram_path)
plt.close()
print(f'误差分布直方图已保存至 {histogram_path}')

print('\n测试完成！')
