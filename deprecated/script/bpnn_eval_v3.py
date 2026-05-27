"""
=============================================================================
模块名称: deprecated/script/bpnn_eval_v3.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (v3 BPNN z-score 评估, 已废弃)
作　　者: TonyNee
创建日期: 2025-11
最后修改: 2026-05-27
=============================================================================

功能概述:
  v3 BPNN 模型评估脚本, 配合 bpnn_train_v3.py。
  加载 Z-Score 归一化参数 (.npz) 和训练好的 v1 BPNN 模型,
  对测试数据进行归一化 -> 推理 -> 反归一化 -> 指标计算。

评估内容:
  - 预测 vs 真实值对比图 (GHz)
  - MAE/MSE/R² 指标
  - 残差曲线图 (MHz)
  - 保存预测结果 CSV

废弃原因: 功能被 root/eval.py 整合, 后者支持 Config 类、分位数误差、
  残差 PDF、分通道 MAE 等更全面的评估。

注意: 此文件已废弃。
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from model import BPNN                                             # v1 BPNN (1-64-32-1)

# =============================================================================
# 1. 设备配置
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: GPU')

# =============================================================================
# 2. 加载测试数据
# =============================================================================
test_csv_path = './data/THz_eval_20250928.csv'
df_test = pd.read_csv(test_csv_path)

x_test_raw = df_test.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)
y_test_raw = df_test.iloc[:, 1].values.astype(np.float32).reshape(-1, 1)

# =============================================================================
# 3. 加载训练时保存的 Z-Score 归一化参数
# =============================================================================
norm_param_path = "./model/norm_params_20250928.npz"
norm = np.load(norm_param_path)

x_mean = norm["x_mean"]                                          # 输入均值
x_std = norm["x_std"]                                            # 输入标准差
y_mean = norm["y_mean"]                                          # 目标均值
y_std = norm["y_std"]                                            # 目标标准差

print("成功加载归一化参数。")

# =============================================================================
# 4. 对测试数据做相同归一化 (必须与训练完全一致)
# =============================================================================
x_test_norm = (x_test_raw - x_mean) / x_std                      # Z-Score 归一化

X_test = torch.tensor(x_test_norm.astype(np.float32)).to(device)
Y_test_raw = y_test_raw                                           # 保留原始 GHz 供对比

# =============================================================================
# 5. 加载模型
# =============================================================================
model = BPNN().to(device)
model_path = './model/bpnn_model_20250928.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f'成功加载模型: {model_path}')

# =============================================================================
# 6. 推理 + 反归一化
# =============================================================================
with torch.no_grad():
    y_pred_norm = model(X_test)                                   # 模型输出 (归一化后)

# 反归一化: Z-Score -> GHz
y_pred = y_pred_norm.cpu().numpy() * y_std + y_mean
y_true = Y_test_raw

# =============================================================================
# 7. 绘制预测对比图 (GHz)
# =============================================================================
plt.figure(figsize=(8, 5))
plt.plot(y_true, label='Standard (True)')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Frequency (GHz)')
plt.title('Test Set: True vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_dir = './result/'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'test_prediction_vs_truth.png'))
plt.close()

# =============================================================================
# 8. 评价指标 (在原始 GHz 尺度上反归一化后计算)
# =============================================================================
mae = np.mean(np.abs(y_pred - y_true))
mse = np.mean((y_pred - y_true) ** 2)
r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

print(f'\n测试集 MAE: {mae*1000:.6f} MHz')
print(f'测试集 MSE: {mse:.12f} (GHz^2)')
print(f'测试集 R²: {r2:.6f}')

# =============================================================================
# 9. 保存预测结果 CSV
# =============================================================================
save_pred_path = os.path.join(output_dir, 'test_predictions.csv')
df_pred = pd.DataFrame({
    'Fexperiment_GHz': x_test_raw.flatten(),                     # 测量频率
    'Fstandard_GHz': y_true.flatten(),                            # 真实频率
    'Fpredicted_GHz': y_pred.flatten()                            # 模型预测频率
})
df_pred.to_csv(save_pred_path, index=False)
print(f'预测结果已保存至 {save_pred_path}')

# =============================================================================
# 10. 绘制残差曲线 (MHz)
# =============================================================================
residuals = (y_true - y_pred) * 1000                              # 转换为 MHz

print(f"残差最大值: {np.max(abs(residuals)):.12f} MHz")
print(f"残差最小值: {np.min(abs(residuals)):.12f} MHz")

plt.figure(figsize=(8, 5))
plt.plot(residuals, label='Residuals', color='orange')
plt.axhline(0, color='black', linestyle='--', linewidth=1)       # 零残差线
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
