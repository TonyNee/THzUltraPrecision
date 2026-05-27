"""
=============================================================================
模块名称: backup/train_resmlp.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (早期备份版本)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  早期简化版 ResMLP 训练脚本。
  使用简单的 train/eval 划分 (无 K-Fold), HuberLoss + AdamW + CosineAnnealingLR,
  带早停机制。

与当前 train.py 的主要差异:
  - 无 K-Fold 交叉验证
  - 无全量数据重训练阶段
  - 硬编码路径和超参数 (不使用 Config 类)
  - 无 CV 结果汇总和可视化

注意: 此文件已废弃, 当前项目使用 train.py 进行训练。
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_resmlp import HighPrecCalibrator                    # 早期独立模型
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 配置 (硬编码, 不使用 Config 类)
# =============================================================================
TRAIN_CSV = '../data/20250928/THz_train_20250928.csv'           # 训练集路径
EVAL_CSV  = '../data/20250928/THz_eval_20250928.csv'            # 验证集路径
SAVE_PATH = './model/resmlp_calibration_best.pth'               # 最佳模型保存路径

BATCH_SIZE = 64                                                  # 批量大小
LR = 1e-3                                                        # 学习率
EPOCHS = 2000                                                    # 最大训练轮数
PATIENCE = 200                                                   # 早停耐心值

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# 1. 数据加载
# =============================================================================
df_train = pd.read_csv(TRAIN_CSV)
df_eval  = pd.read_csv(EVAL_CSV)

# CSV 第一列: 测量频率 (测试值), 第二列: 真实频率 (理论值)
x_train = df_train.iloc[:, 0].values.astype(np.float32)
y_train = df_train.iloc[:, 1].values.astype(np.float32)

x_eval = df_eval.iloc[:, 0].values.astype(np.float32)
y_eval = df_eval.iloc[:, 1].values.astype(np.float32)

# 转换为 PyTorch 张量 (格式: Nx1)
X_train = torch.tensor(x_train.reshape(-1, 1)).to(device)
Y_train = torch.tensor(y_train.reshape(-1, 1)).to(device)

X_eval = torch.tensor(x_eval.reshape(-1, 1)).to(device)
Y_eval = torch.tensor(y_eval.reshape(-1, 1)).to(device)

# 构造 DataLoader
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
eval_loader  = DataLoader(TensorDataset(X_eval, Y_eval), batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# 2. 模型定义
# =============================================================================
model = HighPrecCalibrator().to(device)

# HuberLoss: 结合 MSE (小误差) 和 MAE (大误差) 的优点, 对异常值鲁棒
criterion = nn.HuberLoss()
# AdamW: Adam + 解耦权重衰减
optimizer = optim.AdamW(model.parameters(), lr=LR)
# CosineAnnealingLR: 余弦退火调度, T_max=200 表示完整余弦周期
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_loss = float('inf')                                         # 最佳验证损失 (初始化为无穷大)
patience_count = 0                                               # 早停计数器

# =============================================================================
# 3. 训练循环
# =============================================================================
for epoch in range(1, EPOCHS + 1):
    # ---- 训练阶段 ----
    model.train()
    train_losses = []

    for Xb, Yb in train_loader:
        pred = model(Xb)                                          # 输出为校正后的频率
        loss = criterion(pred, Yb)

        optimizer.zero_grad()                                     # 清零梯度
        loss.backward()                                           # 反向传播
        optimizer.step()                                          # 更新权重
        train_losses.append(loss.item())

    scheduler.step()                                              # 更新学习率

    # ---- 验证阶段 ----
    model.eval()
    with torch.no_grad():                                         # 关闭梯度, 加速推理
        val_losses = []
        for Xe, Ye in eval_loader:
            pred = model(Xe)
            loss = criterion(pred, Ye)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    print(f"Epoch {epoch}/{EPOCHS} | train={train_loss:.6e} | val={val_loss:.6e}")

    # ---- 早停 & 模型保存 ----
    if val_loss < best_loss:
        best_loss = val_loss
        patience_count = 0                                        # 重置计数器

        os.makedirs("./model", exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)                  # 保存最佳权重
    else:
        patience_count += 1

    if patience_count > PATIENCE:
        print("\nEarly stopping triggered!")
        break

print(f"\n训练完成, 最佳模型已保存: {SAVE_PATH}")
