"""
=============================================================================
模块名称: deprecated/script/bpnn_train_v2.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (v2 BPNN min-max, 已废弃)
作　　者: TonyNee
创建日期: 2025-11
最后修改: 2026-05-27
=============================================================================

功能概述:
  v2 BPNN 训练脚本, 使用 Min-Max 归一化 (缩放到 [0, 1] 范围)。
  基于 model_paper.py 的论文 BPNN 结构 (500-50-10), 使用 MSE + Adam。

与 v1 的主要差异:
  - 新增 Min-Max 归一化 (输入/输出都缩放到 [0, 1])
  - 保存归一化参数到 .npy 文件供评估时反归一化
  - 使用 model_paper.py (论文网络, 500-50-10)
  - 损失函数: MSE (非 Huber)

废弃原因: Min-Max 归一化对异常值敏感, 被 v3 (z-score) 取代。

注意: 此文件已废弃。
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

from model_paper import BPNN                                          # 论文 BPNN (500-50-10)

# =============================================================================
# 1. 设备配置
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# =============================================================================
# 2. 加载数据并做 Min-Max 归一化
# =============================================================================
csv_path = './data/train_20251118.csv'
df = pd.read_csv(csv_path)

# 原始数据 (MHz 单位)
x_raw = df['Fexperiment_MHz'].values.astype(np.float32).reshape(-1, 1)
y_raw = df['Fstandard_MHz'].values.astype(np.float32).reshape(-1, 1)

# Min-Max 归一化: 缩放到 [0, 1] 范围
x_min, x_max = x_raw.min(), x_raw.max()
y_min, y_max = y_raw.min(), y_raw.max()

x_data = (x_raw - x_min) / (x_max - x_min)                       # 输入归一化
y_data = (y_raw - y_min) / (y_max - y_min)                       # 目标归一化

# 保存归一化参数 (评估时用于反归一化)
norm_params = {
    'x_min': x_min, 'x_max': x_max,
    'y_min': y_min, 'y_max': y_max
}
np.save('./model/normalization_params.npy', norm_params)

print(f'数据归一化完成:')
print(f'x_data范围: {x_raw.min():.2f} - {x_raw.max():.2f} MHz -> {x_data.min():.6f} - {x_data.max():.6f}')
print(f'y_data范围: {y_raw.min():.2f} - {y_raw.max():.2f} MHz -> {y_data.min():.6f} - {y_data.max():.6f}')

# =============================================================================
# 3. 划分训练集与测试集 (80/20)
# =============================================================================
X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train_np).to(device)
Y_train = torch.tensor(Y_train_np).to(device)
X_test = torch.tensor(X_test_np).to(device)
Y_test = torch.tensor(Y_test_np).to(device)

# =============================================================================
# 4. 模型定义 (论文 BPNN + MSE + Adam)
# =============================================================================
model = BPNN().to(device)
criterion = nn.MSELoss()                                          # 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================================================================
# 5. 训练循环
# =============================================================================
epochs = 1000
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.12f}')

# =============================================================================
# 6. 保存训练损失曲线
# =============================================================================
os.makedirs('./result', exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title('Training Loss (Huber Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/loss_curve.png')
plt.close()

# =============================================================================
# 7. 测试集预测与可视化 (反归一化到原始 MHz)
# =============================================================================
model.eval()
with torch.no_grad():
    y_pred_test_norm = model(X_test)

# 反归一化: [0, 1] -> MHz
y_pred_test = y_pred_test_norm.cpu().numpy() * (y_max - y_min) + y_min
Y_test_original = Y_test_np * (y_max - y_min) + y_min

plt.figure(figsize=(8, 5))
plt.plot(Y_test_original, label='Standard (True)')
plt.plot(y_pred_test, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Frequency (MHz)')
plt.title('Test Set: True vs Predicted (Original Scale)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/test_prediction_vs_truth.png')
plt.close()

# =============================================================================
# 8. 评价指标 (在原始 MHz 尺度上)
# =============================================================================
model.eval()
with torch.no_grad():
    y_pred_train_norm = model(X_train)
    y_pred_train_original = y_pred_train_norm.cpu().numpy() * (y_max - y_min) + y_min
    Y_train_original = Y_train_np * (y_max - y_min) + y_min

# 残差计算
residuals_train = Y_train_original - y_pred_train_original
residuals_test = Y_test_original - y_pred_test

train_mse = np.mean(residuals_train ** 2)
test_mse = np.mean(residuals_test ** 2)

train_r2 = 1 - np.sum(residuals_train ** 2) / np.sum((Y_train_original - np.mean(Y_train_original)) ** 2)
test_r2 = 1 - np.sum(residuals_test ** 2) / np.sum((Y_test_original - np.mean(Y_test_original)) ** 2)

mae = np.mean(np.abs(residuals_test))
std_dev = np.std(residuals_test)

print('\n===== 模型评价指标 (原始尺度) =====')
print(f'训练集 MSE: {train_mse:.6f} MHz²')
print(f'训练集 R²: {train_r2:.6f}')
print(f'测试集 MSE: {test_mse:.6f} MHz²')
print(f'测试集 R²: {test_r2:.6f}')
print(f"残差统计: MAE={mae:.6f} MHz, 标准差={std_dev:.6f} MHz")

# =============================================================================
# 9. 保存模型和归一化参数
# =============================================================================
os.makedirs('./model', exist_ok=True)
torch.save(model.state_dict(), './model/bpnn_model_20250928.pth')
print('模型已保存至 ./model/bpnn_model_20250928.pth')
print('归一化参数已保存至 ./model/normalization_params.npy')
