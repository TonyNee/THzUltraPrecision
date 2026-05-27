"""
=============================================================================
模块名称: deprecated/script/bpnn_train_v3.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (v3 BPNN z-score, 已废弃)
作　　者: TonyNee
创建日期: 2025-11
最后修改: 2026-05-27
=============================================================================

功能概述:
  v3 BPNN 训练脚本, 使用 Z-Score 归一化 (减均值除标准差)。
  基于 model.py 的 v1 BPNN 结构 (1-64-32-1), 使用 HuberLoss + AdamW。

与 v2 的主要差异:
  - 归一化方式: Min-Max -> Z-Score (更稳健)
  - 模型: model_paper (论文 BPNN) -> model (v1 简单 BPNN)
  - 损失函数: MSE -> HuberLoss (更鲁棒)
  - 优化器: Adam -> AdamW (带解耦权重衰减)
  - 归一化参数保存: .npy -> .npz (更标准)

废弃原因: 被 root/train.py 中统一的 K-Fold CV + Config 架构取代。

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

from model import BPNN                                              # v1 BPNN (1-64-32-1)

# =============================================================================
# 1. 设备配置
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: GPU')

# =============================================================================
# 2. 数据加载 + Z-Score 归一化
# =============================================================================
csv_path = './data/THz_train_20250928.csv'
df = pd.read_csv(csv_path)

x_data = df['Fexperiment_GHz'].values.astype(np.float32).reshape(-1, 1)
y_data = df['Fstandard_GHz'].values.astype(np.float32).reshape(-1, 1)

# Z-Score 归一化: (x - mean) / std
x_mean, x_std = x_data.mean(), x_data.std()
y_mean, y_std = y_data.mean(), y_data.std()

x_norm = (x_data - x_mean) / x_std
y_norm = (y_data - y_mean) / y_std

# 保存归一化参数到 .npz (比 .npy 更适合多参数)
np.savez("./model/norm_params_20250928.npz",
         x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

# =============================================================================
# 3. 划分训练集与验证集 (80/20)
# =============================================================================
X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(
    x_norm, y_norm, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train_np).to(device)
Y_train = torch.tensor(Y_train_np).to(device)
X_test = torch.tensor(X_test_np).to(device)
Y_test = torch.tensor(Y_test_np).to(device)

# =============================================================================
# 4. 模型: BPNN + HuberLoss + AdamW
# =============================================================================
model = BPNN().to(device)
criterion = nn.HuberLoss(delta=0.001)                             # delta=0.001: 小误差MSE, 大误差MAE
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

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
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.8f}')

# =============================================================================
# 6. 保存训练损失曲线
# =============================================================================
os.makedirs('./result', exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title('Training Loss (Huber)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/loss_curve.png')
plt.close()

# =============================================================================
# 7. 验证集预测 (反归一化到 GHz)
# =============================================================================
model.eval()
with torch.no_grad():
    y_pred_norm = model(X_test)

# 反归一化: Z-Score -> GHz
Y_test_GHz = Y_test.cpu().numpy() * y_std + y_mean
y_pred_GHz = y_pred_norm.cpu().numpy() * y_std + y_mean

# =============================================================================
# 8. 绘制验证集预测对比图
# =============================================================================
plt.figure(figsize=(8, 5))
plt.plot(Y_test_GHz, label='Standard (True)')
plt.plot(y_pred_GHz, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Frequency (GHz)')
plt.title('Test Set: True vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/test_prediction_vs_truth.png')
plt.close()

# =============================================================================
# 9. 评价指标 (GHz / MHz)
# =============================================================================
with torch.no_grad():
    y_pred_train_norm = model(X_train)
    y_pred_train_GHz = y_pred_train_norm.cpu().numpy() * y_std + y_mean
    Y_train_GHz = Y_train.cpu().numpy() * y_std + y_mean

    # ---- 训练集指标 ----
    train_residuals_MHz = (Y_train_GHz - y_pred_train_GHz) * 1000
    train_mae = np.mean(np.abs(train_residuals_MHz))
    train_std = np.std(train_residuals_MHz)
    train_mse = np.mean((Y_train_GHz - y_pred_train_GHz) ** 2)
    train_r2 = 1 - (
        np.sum((Y_train_GHz - y_pred_train_GHz)**2)
        / np.sum((Y_train_GHz - np.mean(Y_train_GHz))**2)
    )

    # ---- 验证集指标 ----
    test_residuals_MHz = (Y_test_GHz - y_pred_GHz) * 1000
    test_mae = np.mean(np.abs(test_residuals_MHz))
    test_std = np.std(test_residuals_MHz)
    test_mse = np.mean((Y_test_GHz - y_pred_GHz)**2)
    test_r2 = 1 - (
        np.sum((Y_test_GHz - y_pred_GHz)**2)
        / np.sum((Y_test_GHz - np.mean(Y_test_GHz))**2)
    )

# 输出指标对比表格
print('\n===== 模型评价指标（GHz / MHz） =====')
metrics_table = pd.DataFrame({
    '指标': ['MAE (MHz)', 'MSE (GHz^2)', 'R²', 'Std (MHz)'],
    '训练集': [train_mae, train_mse, train_r2, train_std],
    '验证集': [test_mae, test_mse, test_r2, test_std]
})
print(metrics_table.to_string(index=False, float_format='{:.6f}'.format))

# =============================================================================
# 10. 保存模型
# =============================================================================
torch.save(model.state_dict(), './model/bpnn_model_20250928.pth')
print('模型已保存至 ./model/bpnn_model_20250928.pth')
