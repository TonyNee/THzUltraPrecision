import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model_highprec import HighPrecNet

# -----------------------
# 1. 路径配置
# -----------------------
TRAIN_CSV = './data/THz_train_20250928.csv'
EVAL_CSV  = './data/THz_eval_20250928.csv'

MODEL_DIR = './model'
RESULT_DIR = './result'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# -----------------------
# 2. 超参数
# -----------------------
EPOCHS = 2000
BATCH_SIZE = 128
LR = 5e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 100
CLIP_NORM = 1.0
HUBER_DELTA = 1e-3

HIDDEN_DIM = 128
N_BLOCKS = 4
DROPOUT = 0.0

# -----------------------
# 3. 固定随机种子
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -----------------------
# 4. 加载训练/验证数据
# -----------------------
df_train = pd.read_csv(TRAIN_CSV)
df_eval = pd.read_csv(EVAL_CSV)

x_train = df_train['Fexperiment_GHz'].values.astype(np.float32).reshape(-1, 1)
y_train = df_train['Fstandard_GHz'].values.astype(np.float32).reshape(-1, 1)

x_eval = df_eval['Fexperiment_GHz'].values.astype(np.float32).reshape(-1, 1)
y_eval = df_eval['Fstandard_GHz'].values.astype(np.float32).reshape(-1, 1)

# -----------------------
# 5. 归一化，基于训练集计算
# -----------------------
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

# 保存归一化参数
np.savez(os.path.join(MODEL_DIR, 'norm_params_highprec.npz'),
         x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

# 应用归一化（训练 + 验证）
x_train_n = (x_train - x_mean) / x_std
y_train_n = (y_train - y_mean) / y_std

x_eval_n = (x_eval - x_mean) / x_std
y_eval_n = (y_eval - y_mean) / y_std

# -----------------------
# 6. 转 tensor
# -----------------------
X_train = torch.tensor(x_train_n).float().to(device)
Y_train = torch.tensor(y_train_n).float().to(device)
X_eval  = torch.tensor(x_eval_n).float().to(device)
Y_eval  = torch.tensor(y_eval_n).float().to(device)

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

# -----------------------
# 7. 模型定义
# -----------------------
model = HighPrecNet(
    hidden_dim=HIDDEN_DIM,
    n_blocks=N_BLOCKS,
    dropout=DROPOUT
).to(device)

criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=PATIENCE,
    min_lr=1e-7
)

print(model)

# -----------------------
# 8. 训练
# -----------------------
best_val = float('inf')
train_losses, eval_losses = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_epoch = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

        optimizer.step()
        train_loss_epoch += loss.item() * xb.size(0)

    train_loss_epoch /= len(train_loader.dataset)
    train_losses.append(train_loss_epoch)

    # ---- 验证（使用 eval.csv）----
    model.eval()
    with torch.no_grad():
        val_pred = model(X_eval)
        val_loss = criterion(val_pred, Y_eval).item()
    eval_losses.append(val_loss)

    scheduler.step(val_loss)

    # 保存最佳模型
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "highprec_model_best.pth"))

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} | Train={train_loss_epoch:.8e} | Eval={val_loss:.8e} | Best={best_val:.8e}")

# 保存最终模型
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "highprec_model_last.pth"))

# -----------------------
# 9. 训练曲线
# -----------------------
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='train_loss')
plt.plot(eval_losses, label='eval_loss')
plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULT_DIR, 'train_eval_loss.png'))
plt.close()

# -----------------------
# 10. 在 eval.csv 上做最终评估
# -----------------------
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "highprec_model_best.pth")))
model.eval()

with torch.no_grad():
    y_eval_pred_n = model(X_eval).cpu().numpy()

# 反归一化
y_true = y_eval
y_pred = y_eval_pred_n * y_std + y_mean

residuals = (y_true - y_pred) * 1000  # MHz

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
