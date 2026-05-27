"""
=============================================================================
模块名称: deprecated/script/bpnn_eval.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (v1 BPNN 评估, 已废弃)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  v1 BPNN 模型评估脚本。使用原始 GHz 数据 (无归一化),
  计算 MAE/MSE/R², 绘制预测对比图和残差图。

废弃原因: 无归一化导致评估结果不可靠, 且功能被 root/eval.py 取代。

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

from model import BPNN                                             # v1 BPNN

# =============================================================================
# 1. 设备配置
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: GPU')

# =============================================================================
# 2. 加载测试数据 (原始 GHz, 无归一化)
# =============================================================================
test_csv_path = './data/THz_eval_20250928.csv'
df_test = pd.read_csv(test_csv_path)

x_test = df_test.iloc[:, 0].values.astype(np.float32)             # 测量频率
y_test = df_test.iloc[:, 1].values.astype(np.float32)             # 真实频率

X_test = torch.tensor(x_test.reshape(-1, 1)).to(device)
Y_test = torch.tensor(y_test.reshape(-1, 1)).to(device)

# =============================================================================
# 3. 加载模型权重
# =============================================================================
model = BPNN().to(device)
model_path = './model/bpnn_model_20250928.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f'成功加载模型: {model_path}')

# =============================================================================
# 4. 模型预测
# =============================================================================
with torch.no_grad():
    y_pred_test = model(X_test)

# 转回 CPU
y_true = Y_test.cpu().numpy()
y_pred = y_pred_test.cpu().numpy()

# =============================================================================
# 5. 绘制预测对比图
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
# 6. 评价指标
# =============================================================================
criterion = nn.MSELoss()
mse = criterion(y_pred_test, Y_test).item()                       # GHz^2
r2 = 1 - (torch.sum((Y_test - y_pred_test) ** 2)
          / torch.sum((Y_test - torch.mean(Y_test)) ** 2)).item()
mae = torch.mean(torch.abs(y_pred_test - Y_test)).item()           # GHz

print(f'\n测试集 MAE: {mae:.6f}')
print(f'测试集 MSE: {mse:.6f}')
print(f'测试集 R²: {r2:.6f}')

# =============================================================================
# 7. 保存预测结果 CSV
# =============================================================================
save_pred_path = os.path.join(output_dir, 'test_predictions.csv')
df_pred = pd.DataFrame({
    'Fexperiment_GHz': x_test,
    'Fstandard_GHz': y_test,
    'Fpredicted_GHz': y_pred.flatten()
})
df_pred.to_csv(save_pred_path, index=False)
print(f'预测结果已保存至 {save_pred_path}')

# =============================================================================
# 8. 绘制残差曲线 (MHz)
# =============================================================================
residuals = (y_true - y_pred) * 1000                              # 转 MHz

print(f"残差最大值: {np.max(abs(residuals)):.12f} MHz")
print(f"残差最小值: {np.min(abs(residuals)):.12f} MHz")

plt.figure(figsize=(8, 5))
plt.plot(residuals, label='Residuals', color='orange')
plt.axhline(0, color='black', linestyle='--', linewidth=1)       # 零残差参考线
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
