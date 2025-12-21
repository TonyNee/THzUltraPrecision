import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_resmlp import HighPrecCalibrator
from torch.utils.data import DataLoader, TensorDataset

# ============================
# 配置
# ============================
TRAIN_CSV = '../data/20250928/THz_train_20250928.csv'
EVAL_CSV  = '../data/20250928/THz_eval_20250928.csv'
SAVE_PATH = './model/resmlp_calibration_best.pth'

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 2000
PATIENCE = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================
# 1. 数据加载
# ============================
df_train = pd.read_csv(TRAIN_CSV)
df_eval  = pd.read_csv(EVAL_CSV)

# 第一列是测试值，第二列是理论值
x_train = df_train.iloc[:, 0].values.astype(np.float32)
y_train = df_train.iloc[:, 1].values.astype(np.float32)

x_eval = df_eval.iloc[:, 0].values.astype(np.float32)
y_eval = df_eval.iloc[:, 1].values.astype(np.float32)

# 张量
X_train = torch.tensor(x_train.reshape(-1, 1)).to(device)
Y_train = torch.tensor(y_train.reshape(-1, 1)).to(device)

X_eval = torch.tensor(x_eval.reshape(-1, 1)).to(device)
Y_eval = torch.tensor(y_eval.reshape(-1, 1)).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
eval_loader  = DataLoader(TensorDataset(X_eval, Y_eval), batch_size=BATCH_SIZE, shuffle=False)

# ============================
# 2. 模型定义
# ============================
model = HighPrecCalibrator().to(device)

criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_loss = float('inf')
patience_count = 0

# ============================
# 3. 开始训练
# ============================
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for Xb, Yb in train_loader:
        pred = model(Xb)              # 输出已经校正后的频率
        loss = criterion(pred, Yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    scheduler.step()

    # ---------- 验证 ----------
    model.eval()
    with torch.no_grad():
        val_losses = []
        for Xe, Ye in eval_loader:
            pred = model(Xe)
            loss = criterion(pred, Ye)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    print(f"Epoch {epoch}/{EPOCHS} | train={train_loss:.6e} | val={val_loss:.6e}")

    # ---------- Early Stopping ----------
    if val_loss < best_loss:
        best_loss = val_loss
        patience_count = 0

        os.makedirs("./model", exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
    else:
        patience_count += 1

    if patience_count > PATIENCE:
        print("\nEarly stopping triggered!")
        break

print(f"\n训练完成，最佳模型已保存: {SAVE_PATH}")
