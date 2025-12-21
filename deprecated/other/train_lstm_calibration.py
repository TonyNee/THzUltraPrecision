import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from lstm_model import LSTMCalibNet
import matplotlib.pyplot as plt

# --------------------------
# 配置
# --------------------------
SEQ_LEN = 16
EPOCHS = 2000
LR = 1e-3
BATCH = 32
HIDDEN = 128

TRAIN_CSV = "./data/THz_train_20250928.csv"

MODEL_DIR = "./model"
RESULT_DIR = "./result"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# --------------------------
# 加载训练数据
# --------------------------
df = pd.read_csv(TRAIN_CSV)
x = df["Fexperiment_GHz"].values.astype(np.float32)
y = df["Fstandard_GHz"].values.astype(np.float32)

res = y - x  # 残差（GHz）
print("Residual range:", res.min(), res.max())

# --------------------------
# 归一化（保存）
# --------------------------
x_mean, x_std = x.mean(), x.std()
r_mean, r_std = res.mean(), res.std()

np.savez("./model/lstm_norm_params.npz",
         x_mean=x_mean, x_std=x_std,
         r_mean=r_mean, r_std=r_std)

# --------------------------
# 构造序列
# --------------------------
def build_sequences(arr_x, arr_r, seq_len):
    Xs, Rs = [], []
    for i in range(len(arr_x) - seq_len):
        Xs.append(arr_x[i:i+seq_len])
        Rs.append(arr_r[i:i+seq_len])
    return np.array(Xs), np.array(Rs)

X_seq, R_seq = build_sequences(x, res, SEQ_LEN)

# 数据归一化
Xn = (X_seq - x_mean) / (x_std + 1e-12)
Rn = (R_seq - r_mean) / (r_std + 1e-12)

# train/val 划分
X_train, X_val, R_train, R_val = train_test_split(Xn, Rn, test_size=0.2, random_state=42)

# 转成 tensor
X_train = torch.tensor(X_train).float().unsqueeze(-1).to(device)
R_train = torch.tensor(R_train).float().unsqueeze(-1).to(device)

X_val = torch.tensor(X_val).float().unsqueeze(-1).to(device)
R_val = torch.tensor(R_val).float().unsqueeze(-1).to(device)

# --------------------------
# 模型
# --------------------------
model = LSTMCalibNet(hidden_dim=HIDDEN).to(device)
criterion = nn.HuberLoss(delta=1e-3)
optimz = optim.AdamW(model.parameters(), lr=LR)

# --------------------------
# 训练循环
# --------------------------
train_losses, val_losses = [], []
best_val = 1e9

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(X_train.size(0))

    epoch_loss = 0.0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        xb, rb = X_train[idx], R_train[idx]

        optimz.zero_grad()
        pred = model(xb)
        loss = criterion(pred, rb)
        loss.backward()
        optimz.step()

        epoch_loss += loss.item() * xb.size(0)

    epoch_loss /= len(perm)
    train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, R_val).item()
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "./model/lstm_calibration_best.pth")

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | train={epoch_loss:.6e} | val={val_loss:.6e}")

# 训练曲线
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig("./result/lstm_train_curve.png")
plt.close()

print("训练完成，最佳模型已保存 model/lstm_calibration_best.pth")
