import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import shutil  # 用于复制模型

from config import Config


# ============================
# 0. 配置环境 & 数据加载
# ============================
Config.init()
Config.save_yaml()
device = torch.device(Config.DEVICE)
print(f"Device: {device}")

# 创建输出目录（确保存在）
os.makedirs(Config.RESULT_SAVE_DIR, exist_ok=True)

# 加载训练数据
df = pd.read_csv(Config.TRAIN_CSV)
x = df.iloc[:, 0].values.astype(np.float32)
y = df.iloc[:, 1].values.astype(np.float32)

# 准备 KFold
kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.CV_SEED)

# 存储各折结果
cv_val_losses = []
fold_models = []
fold_best_epochs = []  # ✅ 新增：记录各 fold 的最佳 epoch


# ============================
# 1. K-Fold 交叉验证主循环
# ============================
print(f"\n{'='*60}")
print(f" 🔄 Starting {Config.K_FOLDS}-Fold Cross-Validation")
print(f"{'='*60}")

for fold, (train_idx, val_idx) in enumerate(kf.split(x), 1):
    print(f"\n 🔁 Fold {fold}/{Config.K_FOLDS}")

    # 划分数据
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    # 转 Tensor
    X_train = torch.tensor(x_train.reshape(-1, 1), device=device)
    Y_train = torch.tensor(y_train.reshape(-1, 1), device=device)
    X_val = torch.tensor(x_val.reshape(-1, 1), device=device)
    Y_val = torch.tensor(y_val.reshape(-1, 1), device=device)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), 
                              batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), 
                            batch_size=Config.BATCH_SIZE, shuffle=False)

    # 初始化模型、优化器等（每折重新初始化！）
    model = Config.MODEL_CLASS().to(device)
    criterion = Config.build_loss()
    optimizer = Config.build_optimizer(model)
    scheduler = Config.build_scheduler(optimizer)

    best_val_loss = float("inf")
    patience_count = 0
    train_loss_curve = []
    val_loss_curve = []
    best_epoch = 0

    # 训练循环
    for epoch in range(1, Config.EPOCHS + 1):
        # ----- 训练 -----
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

        # ----- 验证 -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xe, Ye in val_loader:
                pred = model(Xe)
                loss = criterion(pred, Ye)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)

        # Early Stopping & Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_epoch = epoch  # ✅ 记录当前最佳 epoch

            # 保存该 fold 最佳模型（带 fold 标识）
            model_path = os.path.join(
                Config.RESULT_SAVE_DIR, f"best_model_fold_{fold}.pth"
            )
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        # 提前停止
        if patience_count >= Config.PATIENCE:
            print(f"  ⏸️ Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6e})")
            break

        # 日志
        if epoch % 100 == 0 or epoch == Config.EPOCHS or patience_count >= Config.PATIENCE - 2:
            print(f"  Epoch {epoch:4d} | train={train_loss:.6e} | val={val_loss:.6e}")

    # 保存该 fold 的 loss 曲线图
    plt.figure(figsize=(8, 4))
    epochs = range(1, len(train_loss_curve) + 1)
    plt.plot(epochs, train_loss_curve, label="Train Loss", alpha=0.8)
    plt.plot(epochs, val_loss_curve, label="Val Loss", alpha=0.8)
    plt.axvline(best_epoch, color='r', linestyle='--', linewidth=0.8, label=f'Best (ep {best_epoch})')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} Loss Curve (Best Val Loss: {best_val_loss:.2e})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, f"loss_fold_{fold}.png"))
    plt.close()

    # 记录结果
    cv_val_losses.append(best_val_loss)
    fold_models.append(model_path)
    fold_best_epochs.append(best_epoch)  # ✅ 保存 best epoch

    print(f"  ✅ Fold {fold} done. Best Val Loss: {best_val_loss:.6e} (epoch {best_epoch})")


# ============================
# 2. 汇总 CV 结果
# ============================
cv_val_losses = np.array(cv_val_losses)
mean_val_loss = cv_val_losses.mean()
std_val_loss = cv_val_losses.std()

print(f"\n{'='*60}")
print(f" 📊 {Config.K_FOLDS}-Fold Cross-Validation Results")
print(f"{'='*60}")
for i, vl in enumerate(cv_val_losses, 1):
    print(f"Fold {i:2d} Val Loss: {vl:.6e} (best epoch: {fold_best_epochs[i-1]})")
print(f"{'-'*60}")
print(f"Mean Val Loss: {mean_val_loss:.6e} ± {std_val_loss:.6e}")
print(f"95% CI: [{mean_val_loss - 1.96*std_val_loss/np.sqrt(Config.K_FOLDS):.6e}, "
      f"{mean_val_loss + 1.96*std_val_loss/np.sqrt(Config.K_FOLDS):.6e}]")

# 保存结果到文本
result_txt = os.path.join(Config.RESULT_SAVE_DIR, "cv_results.txt")
with open(result_txt, "w") as f:
    f.write(f"{Config.K_FOLDS}-Fold CV Results\n")
    f.write("="*60 + "\n")
    for i, (vl, ep) in enumerate(zip(cv_val_losses, fold_best_epochs), 1):
        f.write(f"Fold {i:2d}: Val Loss = {vl:.6e}, Best Epoch = {ep}\n")
    f.write("-"*60 + "\n")
    f.write(f"Mean Val Loss: {mean_val_loss:.6e}\n")
    f.write(f"Std:            {std_val_loss:.6e}\n")
    f.write(f"Avg Best Epoch: {int(np.mean(fold_best_epochs))}\n")

print(f"\n✅ CV results saved to: {result_txt}")


# ============================
# 3. 复制 CV 最佳模型（可选备份）
# ============================
best_cv_fold_idx = int(np.argmin(cv_val_losses))  # 0-based
best_cv_model_path = fold_models[best_cv_fold_idx]
best_cv_epoch = fold_best_epochs[best_cv_fold_idx]

cv_best_save_path = os.path.join(Config.RESULT_SAVE_DIR, "best_model_cv.pth")
shutil.copy(best_cv_model_path, cv_best_save_path)
print(f"\n📥 CV best model (Fold {best_cv_fold_idx+1}, epoch {best_cv_epoch}) "
      f"copied to: {cv_best_save_path}")


# ============================
# 4. 全量数据重新训练（Refit on Full Data）
# ============================
print(f"\n{'='*60}")
print(" 🔁 Retraining on Full Dataset")
print(f"{'='*60}")

# 策略：用各 fold 最佳 epoch 的平均值（避免过拟合）
retrain_epochs = int(np.round(np.mean(fold_best_epochs)))
print(f"📈 Retrain Epochs = avg(best epochs) = {np.mean(fold_best_epochs):.1f} → {retrain_epochs}")

# 准备全量数据
X_full = torch.tensor(x.reshape(-1, 1), device=device)
Y_full = torch.tensor(y.reshape(-1, 1), device=device)
full_loader = DataLoader(
    TensorDataset(X_full, Y_full),
    batch_size=Config.BATCH_SIZE,
    shuffle=True
)

# 初始化最终模型
final_model = Config.MODEL_CLASS().to(device)
criterion = Config.build_loss()
optimizer = Config.build_optimizer(final_model)
scheduler = Config.build_scheduler(optimizer)

# 全量训练
print(f"🚀 Training on full data ({len(x)} samples) for {retrain_epochs} epochs...")
train_loss_curve_full = []

for epoch in range(1, retrain_epochs + 1):
    final_model.train()
    batch_losses = []
    
    for Xb, Yb in full_loader:
        pred = final_model(Xb)
        loss = criterion(pred, Yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    if scheduler is not None:
        scheduler.step()
    
    epoch_loss = np.mean(batch_losses)
    train_loss_curve_full.append(epoch_loss)
    
    if epoch % 50 == 0 or epoch == retrain_epochs:
        print(f"  Epoch {epoch:4d}/{retrain_epochs} | train_loss = {epoch_loss:.6e}")

# 保存最终模型到标准路径（Config.MODEL_SAVE_PATH）
torch.save(final_model.state_dict(), Config.MODEL_SAVE_PATH)
print(f"\n✅ Final model (full-data retrain) saved to: {Config.MODEL_SAVE_PATH}")

# 保存全量训练 loss 曲线
plt.figure(figsize=(8, 4))
epochs = range(1, len(train_loss_curve_full) + 1)
plt.plot(epochs, train_loss_curve_full, label="Full Train Loss", color='purple', linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Full-Data Training (Epochs={retrain_epochs})")
plt.grid(True)
plt.legend()
plt.tight_layout()
full_loss_fig = os.path.join(Config.RESULT_SAVE_DIR, "loss_full_train.png")
plt.savefig(full_loss_fig)
plt.close()
print(f"📊 Full-train loss curve saved to: {full_loss_fig}")


# ============================
# 5. 总结
# ============================
print(f"\n{'='*60}")
print(" ✅ Training Pipeline Completed!")
print(f"{'='*60}")
print(f"• CV Mean Val Loss: {mean_val_loss:.6e} ± {std_val_loss:.6e}")
print(f"• Final Model (full retrain): {Config.MODEL_SAVE_PATH}")
print(f"• CV Best Model (backup):      {cv_best_save_path}")
print(f"• Results Directory:           {Config.RESULT_SAVE_DIR}")
print(f"{'='*60}")





