"""
=============================================================================
模块名称: deprecated/other/train_lstm_calibration.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (LSTM 实验, 已废弃)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  LSTM 频率校准网络的训练脚本。
  将频率数据组织为长度为 16 的滑动窗口序列, 归一化后输入 LSTM,
  学习预测频率残差 (修正量)。

训练策略:
  - 序列构造: 滑动窗口 (seq_len=16), 前后重叠
  - 数据归一化: z-score (减均值除标准差), 归一化参数保存在 .npz 文件
  - 损失函数: HuberLoss (delta=1e-3, 鲁棒回归)
  - 优化器: AdamW (lr=1e-3)
  - 无学习率调度器, 无早停, 始终训练满 2000 轮

废弃原因: 频率校准是点对点映射, 不需要序列建模。

注意: 此文件已废弃。
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from lstm_model import LSTMCalibNet
import matplotlib.pyplot as plt

# =============================================================================
# 配置 (硬编码)
# =============================================================================
SEQ_LEN = 16                                                     # 滑动窗口长度 (序列长度)
EPOCHS = 2000                                                    # 训练轮数 (无早停)
LR = 1e-3                                                        # 学习率
BATCH = 32                                                       # 批量大小
HIDDEN = 128                                                     # LSTM 隐藏层维度

TRAIN_CSV = "./data/THz_train_20250928.csv"

MODEL_DIR = "./model"
RESULT_DIR = "./result"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# =============================================================================
# 1. 加载训练数据
# =============================================================================
df = pd.read_csv(TRAIN_CSV)
x = df["Fexperiment_GHz"].values.astype(np.float32)              # 实验测量频率
y = df["Fstandard_GHz"].values.astype(np.float32)                # 标准真实频率

# 残差 = 标准值 - 实验值 (GHz), 即需要修正的量
res = y - x
print("Residual range:", res.min(), res.max())

# =============================================================================
# 2. 数据归一化: z-score (保存参数供评估时使用)
# =============================================================================
x_mean, x_std = x.mean(), x.std()                                # 频率均值/标准差
r_mean, r_std = res.mean(), res.std()                            # 残差均值/标准差

# 保存归一化参数到 .npz 文件
np.savez("./model/lstm_norm_params.npz",
         x_mean=x_mean, x_std=x_std,
         r_mean=r_mean, r_std=r_std)

# =============================================================================
# 3. 构造滑动窗口序列
# =============================================================================
def build_sequences(arr_x, arr_r, seq_len):
    """
    将一维数组转换为重叠的滑动窗口序列

    参数:
      arr_x:   频率数组 (N,)
      arr_r:   残差数组 (N,)
      seq_len: 窗口长度

    返回:
      Xs: (N - seq_len, seq_len) 频率序列
      Rs: (N - seq_len, seq_len) 残差序列
    """
    Xs, Rs = [], []
    for i in range(len(arr_x) - seq_len):
        Xs.append(arr_x[i:i+seq_len])                            # 窗口 [i, i+seq_len)
        Rs.append(arr_r[i:i+seq_len])
    return np.array(Xs), np.array(Rs)

X_seq, R_seq = build_sequences(x, res, SEQ_LEN)

# ---- 归一化 ----
Xn = (X_seq - x_mean) / (x_std + 1e-12)                          # 频率 z-score
Rn = (R_seq - r_mean) / (r_std + 1e-12)                          # 残差 z-score (1e-12 防除零)

# ---- 训练/验证划分 (80/20) ----
X_train, X_val, R_train, R_val = train_test_split(Xn, Rn, test_size=0.2, random_state=42)

# ---- 转为 PyTorch 张量 ----
# unsqueeze(-1): 从 (batch, seq_len) 变为 (batch, seq_len, 1) 以匹配 LSTM 输入格式
X_train = torch.tensor(X_train).float().unsqueeze(-1).to(device)
R_train = torch.tensor(R_train).float().unsqueeze(-1).to(device)

X_val = torch.tensor(X_val).float().unsqueeze(-1).to(device)
R_val = torch.tensor(R_val).float().unsqueeze(-1).to(device)

# =============================================================================
# 4. 模型初始化
# =============================================================================
model = LSTMCalibNet(hidden_dim=HIDDEN).to(device)

# HuberLoss: delta=1e-3, 对小误差用 MSE (精确), 对大误差用 MAE (鲁棒)
criterion = nn.HuberLoss(delta=1e-3)
# AdamW: 解耦权重衰减的 Adam 变体
optimz = optim.AdamW(model.parameters(), lr=LR)

# =============================================================================
# 5. 训练循环 (无早停, 始终满 2000 轮)
# =============================================================================
train_losses, val_losses = [], []
best_val = 1e9                                                   # 最佳验证损失

for epoch in range(1, EPOCHS + 1):
    # ---- 训练阶段: 随机打乱批次顺序 ----
    model.train()
    perm = torch.randperm(X_train.size(0))                       # 随机排列索引

    epoch_loss = 0.0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]                                    # 取一个 batch 的索引
        xb, rb = X_train[idx], R_train[idx]

        optimz.zero_grad()
        pred = model(xb)
        loss = criterion(pred, rb)
        loss.backward()
        optimz.step()

        epoch_loss += loss.item() * xb.size(0)                   # 累积损失 (加权)

    epoch_loss /= len(perm)                                      # 平均损失
    train_losses.append(epoch_loss)

    # ---- 验证阶段 ----
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, R_val).item()
    val_losses.append(val_loss)

    # 保存最佳模型
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "./model/lstm_calibration_best.pth")

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | train={epoch_loss:.6e} | val={val_loss:.6e}")

# =============================================================================
# 6. 保存训练曲线 (对数坐标)
# =============================================================================
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.yscale("log")                                                # 对数坐标更好观察后期收敛
plt.legend()
plt.grid()
plt.savefig("./result/lstm_train_curve.png")
plt.close()

print("训练完成, 最佳模型已保存 model/lstm_calibration_best.pth")
