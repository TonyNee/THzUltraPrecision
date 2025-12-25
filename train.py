import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config


# ============================
# 0. 配置环境
# ============================
Config.init()
Config.save_yaml() 
device = torch.device(Config.DEVICE)
print(f"Device: {device}")


# ============================
# 1. 数据加载
# ============================
df_train = pd.read_csv(Config.TRAIN_CSV)
df_eval  = pd.read_csv(Config.EVAL_CSV)

x_train = df_train.iloc[:, 0].values.astype(np.float32)
y_train = df_train.iloc[:, 1].values.astype(np.float32)

x_eval = df_eval.iloc[:, 0].values.astype(np.float32)
y_eval = df_eval.iloc[:, 1].values.astype(np.float32)

X_train = torch.tensor(x_train.reshape(-1, 1), device=device)
Y_train = torch.tensor(y_train.reshape(-1, 1), device=device)

X_eval = torch.tensor(x_eval.reshape(-1, 1), device=device)
Y_eval = torch.tensor(y_eval.reshape(-1, 1), device=device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(TensorDataset(X_eval, Y_eval), batch_size=Config.BATCH_SIZE, shuffle=False)


# ============================
# 2. 模型定义
# ============================
model = Config.MODEL_CLASS().to(device)

criterion = Config.build_loss()
optimizer = Config.build_optimizer(model)
scheduler = Config.build_scheduler(optimizer)

best_loss = float("inf")
patience_count = 0


# ============================
# 3. 开始训练
# ============================
train_loss_curve = []
val_loss_curve = []
for epoch in range(1, Config.EPOCHS + 1):
    model.train()
    train_losses = []

    for Xb, Yb in train_loader:
        pred = model(Xb)
        loss = criterion(pred, Yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    if scheduler is not None:
        scheduler.step()

    # ---------- 验证 ----------
    model.eval()
    val_losses = []
    with torch.no_grad():
        for Xe, Ye in eval_loader:
            pred = model(Xe)
            loss = criterion(pred, Ye)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    train_loss_curve.append(train_loss)
    val_loss_curve.append(val_loss)

    if epoch % 100 == 0 or epoch == Config.EPOCHS or (patience_count >= Config.PATIENCE):
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"train={train_loss:.6e} | val={val_loss:.6e}"
        )

    # ---------- Early Stopping ----------
    if val_loss < best_loss:
        best_loss = val_loss
        patience_count = 0

        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    else:
        patience_count += 1

    if patience_count >= Config.PATIENCE:
        print("\nEarly stopping triggered!")
        break

print(f"\n训练完成，最佳模型已保存: {Config.MODEL_SAVE_PATH}")

epochs = range(1, len(train_loss_curve) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss_curve, label="Train Loss")
plt.plot(epochs, val_loss_curve, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{Config.MODEL_TYPE.upper()} Training Curve")
plt.legend()
plt.grid(True)
loss_fig_path = os.path.join(Config.RESULT_SAVE_DIR, "loss_curve.png")
plt.tight_layout()
plt.savefig(loss_fig_path)
plt.close()

print(f"Loss curve saved to: {loss_fig_path}")


