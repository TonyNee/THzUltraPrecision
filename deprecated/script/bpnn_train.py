"""
=============================================================================
模块名称: deprecated/script/bpnn_train.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (v1 BPNN 训练, 已废弃)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  v1 BPNN 训练脚本。使用原始 GHz 数据 (无归一化), HuberLoss + AdamW,
  80/20 训练/测试划分。

特点:
  - 无数据归一化 (直接使用原始 GHz)
  - 无 K-Fold CV
  - 无验证集监控/早停
  - 训练全程使用全量数据 (无 mini-batch)

废弃原因: 归一化缺失导致训练不稳定, 被 v2 (min-max) / v3 (z-score) 取代。

注意: 此文件已废弃, 使用 deprecated/script/model.py (v1 BPNN)。
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

from model import BPNN                                             # v1 BPNN 模型

# =============================================================================
# 1. 设备配置
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: GPU')

# =============================================================================
# 2. 加载原始数据 (无归一化)
# =============================================================================
csv_path = './data/THz_train_20250928.csv'
df = pd.read_csv(csv_path)

# 直接使用原始 GHz 值
x_data = df['Fexperiment_GHz'].values.astype(np.float32).reshape(-1, 1)
y_data = df['Fstandard_GHz'].values.astype(np.float32).reshape(-1, 1)

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
# 4. 模型定义
# =============================================================================
model = BPNN().to(device)

# HuberLoss: delta=0.001, 小误差用 MSE, 大误差用 MAE (鲁棒)
criterion = nn.HuberLoss(delta=0.001)
# AdamW: 带解耦权重衰减的 Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# =============================================================================
# 5. 训练循环 (无 mini-batch, 全量数据梯度下降)
# =============================================================================
epochs = 1000                                                    # 固定 1000 轮
losses = []                                                      # 记录每轮损失

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.6f}')

# =============================================================================
# 6. 训练损失曲线
# =============================================================================
os.makedirs('./result', exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title('Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/loss_curve.png')
plt.close()

# =============================================================================
# 7. 验证集预测与可视化
# =============================================================================
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)

# 转回 CPU numpy
Y_test_np = Y_test.cpu().numpy()
y_pred_np = y_pred_test.cpu().numpy()

plt.figure(figsize=(8, 5))
plt.plot(Y_test_np, label='Standard (True)')
plt.plot(y_pred_np, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Frequency (GHz)')
plt.title('Test Set: True vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/test_prediction_vs_truth.png')
plt.close()

# =============================================================================
# 8. 评价指标 (训练集 + 验证集)
# =============================================================================
model.eval()
with torch.no_grad():
    # 训练集预测
    y_pred_train = model(X_train)

    # 训练集指标
    train_mse = criterion(y_pred_train, Y_train).item()
    train_r2 = 1 - (torch.sum((Y_train - y_pred_train) ** 2)
                    / torch.sum((Y_train - torch.mean(Y_train)) ** 2)).item()

    # 训练集残差统计 (MHz)
    train_residuals = (Y_train.cpu().numpy() - y_pred_train.cpu().numpy()) * 1000
    train_mae = np.mean(np.abs(train_residuals))
    train_std = np.std(train_residuals)

    # 验证集预测
    y_pred_test = model(X_test)

    # 验证集指标
    test_mse = criterion(y_pred_test, Y_test).item()
    test_r2 = 1 - (torch.sum((Y_test - y_pred_test) ** 2)
                   / torch.sum((Y_test - torch.mean(Y_test)) ** 2)).item()

    # 验证集残差统计 (MHz)
    test_residuals = (Y_test.cpu().numpy() - y_pred_test.cpu().numpy()) * 1000
    test_mae = np.mean(np.abs(test_residuals))
    test_std = np.std(test_residuals)

# 打印指标对比表格
print('\n===== 模型评价指标 =====')
metrics_table = pd.DataFrame({
    '指标': ['MAE', 'MSE', 'R²', 'S²'],
    '训练集': [train_mae, train_mse, train_r2, train_std],
    '验证集': [test_mae, test_mse, test_r2, test_std]
})
print(metrics_table.to_string(index=False, float_format='{:.6f}'.format))

# =============================================================================
# 9. 保存模型
# =============================================================================
torch.save(model.state_dict(), './model/bpnn_model_20250928.pth')
print('模型已保存至 ./model/bpnn_model_20250928.pth')
