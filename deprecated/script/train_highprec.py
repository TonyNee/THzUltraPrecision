"""
=============================================================================
模块名称: deprecated/script/train_highprec.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (HighPrecNet, 已废弃)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  HighPrecNet 残差网络训练脚本。
  使用 z-score 归一化、HuberLoss、AdamW + ReduceLROnPlateau 调度器,
  带梯度裁剪和早停机制。

训练策略:
  - 数据归一化: z-score (保存参数到 .npz)
  - 梯度裁剪: clip_grad_norm_=1.0 (防止梯度爆炸)
  - 学习率调度: ReduceLROnPlateau (factor=0.5, patience=100, min_lr=1e-7)
  - 保存最佳模型 (基于验证损失) 和最终模型

废弃原因: 依赖独立的 model_highprec.py (已不存在于当前项目),
  功能被 root/train.py 的 K-Fold CV 流程取代。

注意: 此文件依赖 model_highprec.py 中的 HighPrecNet 类。
=============================================================================
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model_highprec import HighPrecNet

# =============================================================================
# 1. 路径配置
# =============================================================================
TRAIN_CSV = './data/THz_train_20250928.csv'
EVAL_CSV  = './data/THz_eval_20250928.csv'

MODEL_DIR = './model'
RESULT_DIR = './result'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# =============================================================================
# 2. 超参数
# =============================================================================
EPOCHS = 2000                                                    # 最大训练轮数
BATCH_SIZE = 128                                                 # 批量大小
LR = 5e-4                                                        # 学习率
WEIGHT_DECAY = 1e-5                                              # AdamW 权重衰减
PATIENCE = 100                                                   # 学习率调度器耐心值
CLIP_NORM = 1.0                                                  # 梯度裁剪阈值
HUBER_DELTA = 1e-3                                               # HuberLoss delta

HIDDEN_DIM = 128                                                 # 隐藏层维度
N_BLOCKS = 4                                                     # 残差块数量
DROPOUT = 0.0                                                    # Dropout 比例

# =============================================================================
# 3. 固定随机种子 (保证可复现)
# =============================================================================
def set_seed(seed=42):
    """固定 Python/NumPy/PyTorch 随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)                          # 多 GPU 场景也固定

set_seed(SEED)

# =============================================================================
# 4. 加载训练/验证数据
# =============================================================================
df_train = pd.read_csv(TRAIN_CSV)
df_eval = pd.read_csv(EVAL_CSV)

x_train = df_train['Fexperiment_GHz'].values.astype(np.float32).reshape(-1, 1)
y_train = df_train['Fstandard_GHz'].values.astype(np.float32).reshape(-1, 1)

x_eval = df_eval['Fexperiment_GHz'].values.astype(np.float32).reshape(-1, 1)
y_eval = df_eval['Fstandard_GHz'].values.astype(np.float32).reshape(-1, 1)

# =============================================================================
# 5. 归一化 (z-score, 基于训练集计算)
# =============================================================================
x_mean, x_std = x_train.mean(), x_train.std()                    # 输入均值/标准差
y_mean, y_std = y_train.mean(), y_train.std()                    # 目标均值/标准差

# 保存归一化参数供评估时使用
np.savez(os.path.join(MODEL_DIR, 'norm_params_highprec.npz'),
         x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

# 应用归一化 (训练 + 验证)
x_train_n = (x_train - x_mean) / x_std
y_train_n = (y_train - y_mean) / y_std

x_eval_n = (x_eval - x_mean) / x_std
y_eval_n = (y_eval - y_mean) / y_std

# =============================================================================
# 6. 转 Tensor & DataLoader
# =============================================================================
X_train = torch.tensor(x_train_n).float().to(device)
Y_train = torch.tensor(y_train_n).float().to(device)
X_eval  = torch.tensor(x_eval_n).float().to(device)
Y_eval  = torch.tensor(y_eval_n).float().to(device)

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

# =============================================================================
# 7. 模型定义
# =============================================================================
model = HighPrecNet(
    hidden_dim=HIDDEN_DIM,
    n_blocks=N_BLOCKS,
    dropout=DROPOUT
).to(device)

# HuberLoss + AdamW + ReduceLROnPlateau
criterion = nn.HuberLoss(delta=HUBER_DELTA)                      # 鲁棒损失
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',                                                   # 监控 val_loss 最小值
    factor=0.5,                                                   # 每次衰减为原来的 50%
    patience=PATIENCE,                                            # PATIENCE 轮无改善则衰减
    min_lr=1e-7                                                   # 学习率下限
)

print(model)

# =============================================================================
# 8. 训练循环
# =============================================================================
best_val = float('inf')                                          # 最佳验证损失
train_losses, eval_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # ---- 训练阶段 ----
    model.train()
    train_loss_epoch = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()

        # 梯度裁剪: 限制梯度 L2 范数不超过 CLIP_NORM, 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

        optimizer.step()
        train_loss_epoch += loss.item() * xb.size(0)

    train_loss_epoch /= len(train_loader.dataset)
    train_losses.append(train_loss_epoch)

    # ---- 验证 ----
    model.eval()
    with torch.no_grad():
        val_pred = model(X_eval)
        val_loss = criterion(val_pred, Y_eval).item()
    eval_losses.append(val_loss)

    # 更新学习率 (基于验证损失)
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "highprec_model_best.pth"))

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} | Train={train_loss_epoch:.8e} | Eval={val_loss:.8e} | Best={best_val:.8e}")

# 保存最终模型
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "highprec_model_last.pth"))

# =============================================================================
# 9. 训练曲线 (对数坐标)
# =============================================================================
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='train_loss')
plt.plot(eval_losses, label='eval_loss')
plt.yscale('log')                                                # 对数坐标更易观察后期收敛
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULT_DIR, 'train_eval_loss.png'))
plt.close()

# =============================================================================
# 10. 在 eval.csv 上做最终评估
# =============================================================================
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "highprec_model_best.pth")))
model.eval()

with torch.no_grad():
    y_eval_pred_n = model(X_eval).cpu().numpy()

# 反归一化: z-score -> GHz
y_true = y_eval
y_pred = y_eval_pred_n * y_std + y_mean

# 残差转为 MHz
residuals = (y_true - y_pred) * 1000

# 标准回归指标
MAE = np.mean(np.abs(residuals))
STD = np.std(residuals)
MSE = np.mean((y_true - y_pred)**2)
R2  = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

print("\n===== 最终评估结果（基于 eval.csv） =====")
print(f"MAE: {MAE:.6f} MHz")
print(f"STD: {STD:.6f} MHz")
print(f"MSE: {MSE:.12f} (GHz^2)")
print(f"R2 : {R2:.6f}")

# 保存评估结果
df_out = pd.DataFrame({
    'x_GHz': x_eval.flatten(),
    'y_true_GHz': y_true.flatten(),
    'y_pred_GHz': y_pred.flatten(),
    'residual_MHz': residuals.flatten()
})
df_out.to_csv(os.path.join(RESULT_DIR, 'eval_predictions_highprec.csv'), index=False)
print("评估结果已保存。")
