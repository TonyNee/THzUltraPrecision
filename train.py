"""
=============================================================================
模块名称: train.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统
作　　者: TonyNee
创建日期: 2025-12-18
最后修改: 2026-05-27
=============================================================================

功能概述:
  本模块实现了完整的 K-Fold 交叉验证训练流程, 包含以下阶段:
  1. 环境初始化 — 从 Config 加载配置, 保存 YAML
  2. K-Fold 交叉验证 — 训练 K 个模型, 每个有独立的早停、loss 曲线图
  3. CV 结果汇总 — 计算平均/标准差/置信区间, 保存到文本文件
  4. 全量数据重训练 — 使用 CV 平均最优 epoch 数, 在全部数据上训练最终模型
  5. 总结输出 — 打印所有关键路径和指标

训练策略:
  - 早停机制 (Early Stopping): 连续 PATIENCE 轮验证损失无改善则停止
  - 每折保存最佳模型 (best_model_fold_{k}.pth)
  - CV 最佳模型复制备份 (best_model_cv.pth)
  - 全量重训练 epoch 数 = ceil(avg(best_epochs)), 避免过拟合
  - 生成 per-fold loss 曲线图 + 全量训练 loss 曲线图

使用方式:
  python train.py
  (需先设置 Config 类中的 MODEL_TYPE, TRAIN_CSV 等参数)
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import shutil                                                # 用于模型文件复制

from config import Config


# =============================================================================
# 阶段 0: 配置环境 & 数据加载
# =============================================================================
# 初始化 Config: 解析模型注册表 -> 填充默认超参数 -> 生成路径 -> 创建目录
Config.init()
Config.save_yaml()                                             # 持久化当前配置供评估时复现

# 确定运算设备 (CUDA 或 CPU)
device = torch.device(Config.DEVICE)
print(f"Device: {device}")

# 确保输出目录存在 (init 已创建, 此处是防御性检查)
os.makedirs(Config.RESULT_SAVE_DIR, exist_ok=True)

# 加载训练数据: CSV 格式为两列 (测量频率, 真实频率)
df = pd.read_csv(Config.TRAIN_CSV)
x = df.iloc[:, 0].values.astype(np.float32)                   # 第一列: 测量频率 (GHz)
y = df.iloc[:, 1].values.astype(np.float32)                   # 第二列: 真实频率 (GHz)

# 初始化 K 折交叉验证划分器
# shuffle=True 保证每折数据分布均匀, random_state 保证可复现
kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.CV_SEED)

# 存储各折结果的容器
cv_val_losses = []                                             # 每折最佳验证损失
fold_models = []                                               # 每折最佳模型路径
fold_best_epochs = []                                          # 每折最佳 epoch (用于全量重训练)


# =============================================================================
# 阶段 1: K-Fold 交叉验证主循环
# =============================================================================
print(f"\n{'='*60}")
print(f" 🔄 Starting {Config.K_FOLDS}-Fold Cross-Validation")
print(f"{'='*60}")

for fold, (train_idx, val_idx) in enumerate(kf.split(x), 1):
    print(f"\n 🔁 Fold {fold}/{Config.K_FOLDS}")

    # ---- 数据划分: 按 KFold 索引分割训练/验证集 ----
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    # ---- 转为 PyTorch 张量并移到目标设备 ----
    X_train = torch.tensor(x_train.reshape(-1, 1), device=device)
    Y_train = torch.tensor(y_train.reshape(-1, 1), device=device)
    X_val = torch.tensor(x_val.reshape(-1, 1), device=device)
    Y_val = torch.tensor(y_val.reshape(-1, 1), device=device)

    # ---- 构造 DataLoader ----
    # 训练集: shuffle=True 打乱顺序, 避免批次间模式偏差
    # 验证集: shuffle=False 保持顺序, 保证结果可复现
    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val),
                            batch_size=Config.BATCH_SIZE, shuffle=False)

    # ---- 初始化模型、损失函数、优化器、调度器 (每折重新初始化!) ----
    model = Config.MODEL_CLASS().to(device)
    criterion = Config.build_loss()
    optimizer = Config.build_optimizer(model)
    scheduler = Config.build_scheduler(optimizer)

    # ---- 早停状态变量 ----
    best_val_loss = float("inf")                                # 当前最佳验证损失 (初始化为无穷大)
    patience_count = 0                                          # 连续无改善计数器
    train_loss_curve = []                                       # 训练损失历史
    val_loss_curve = []                                         # 验证损失历史
    best_epoch = 0                                              # 达到最佳损失的 epoch

    # ---- 训练循环 ----
    for epoch in range(1, Config.EPOCHS + 1):
        # ========== 训练阶段 ==========
        model.train()
        train_losses = []
        for Xb, Yb in train_loader:
            pred = model(Xb)                                    # 前向传播
            loss = criterion(pred, Yb)                          # 计算损失

            optimizer.zero_grad()                               # 清零梯度缓存
            loss.backward()                                     # 反向传播
            optimizer.step()                                    # 更新参数
            train_losses.append(loss.item())

        # 更新学习率调度器 (每 epoch 一步)
        if scheduler is not None:
            scheduler.step()

        # ========== 验证阶段 ==========
        model.eval()
        val_losses = []
        with torch.no_grad():                                   # 关闭梯度计算, 节省显存
            for Xe, Ye in val_loader:
                pred = model(Xe)
                loss = criterion(pred, Ye)
                val_losses.append(loss.item())

        # 计算 epoch 平均损失
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)

        # ---- 早停判断 & 模型保存 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0                                  # 重置计数器
            best_epoch = epoch

            # 保存该 fold 最佳模型 (文件名含 fold 编号)
            model_path = os.path.join(
                Config.RESULT_SAVE_DIR, f"best_model_fold_{fold}.pth"
            )
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        # 达到耐心上限, 提前停止
        if patience_count >= Config.PATIENCE:
            print(f"  ⏸️ Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6e})")
            break

        # 每 100 轮或最后阶段打印日志
        if epoch % 100 == 0 or epoch == Config.EPOCHS or patience_count >= Config.PATIENCE - 2:
            print(f"  Epoch {epoch:4d} | train={train_loss:.6e} | val={val_loss:.6e}")

    # ---- 保存该 fold 的 loss 曲线图 ----
    plt.figure(figsize=(8, 4))
    epochs = range(1, len(train_loss_curve) + 1)
    plt.plot(epochs, train_loss_curve, label="Train Loss", alpha=0.8)
    plt.plot(epochs, val_loss_curve, label="Val Loss", alpha=0.8)
    plt.axvline(best_epoch, color='r', linestyle='--', linewidth=0.8,
                label=f'Best (ep {best_epoch})')               # 红色虚线标注最佳 epoch
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} Loss Curve (Best Val Loss: {best_val_loss:.2e})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, f"loss_fold_{fold}.png"))
    plt.close()

    # 记录本折结果
    cv_val_losses.append(best_val_loss)
    fold_models.append(model_path)
    fold_best_epochs.append(best_epoch)

    print(f"  ✅ Fold {fold} done. Best Val Loss: {best_val_loss:.6e} (epoch {best_epoch})")


# =============================================================================
# 阶段 2: 汇总 CV 结果
# =============================================================================
cv_val_losses = np.array(cv_val_losses)
mean_val_loss = cv_val_losses.mean()                            # 平均验证损失
std_val_loss = cv_val_losses.std()                              # 标准差

# ---- 打印 CV 汇总 ----
print(f"\n{'='*60}")
print(f" 📊 {Config.K_FOLDS}-Fold Cross-Validation Results")
print(f"{'='*60}")
for i, vl in enumerate(cv_val_losses, 1):
    print(f"Fold {i:2d} Val Loss: {vl:.6e} (best epoch: {fold_best_epochs[i-1]})")
print(f"{'-'*60}")
print(f"Mean Val Loss: {mean_val_loss:.6e} ± {std_val_loss:.6e}")

# 95% 置信区间 (基于 t 分布的近似, K 较小时较宽)
n_folds = Config.K_FOLDS
ci_half = 1.96 * std_val_loss / np.sqrt(n_folds)               # 1.96 = z_{0.025} (正态近似)
print(f"95% CI: [{mean_val_loss - ci_half:.6e}, "
      f"{mean_val_loss + ci_half:.6e}]")

# ---- 保存 CV 结果到文本文件 ----
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


# =============================================================================
# 阶段 3: 复制 CV 最佳模型 (可选备份)
# =============================================================================
# 在所有折中选择验证损失最小的一折, 复制其模型作为 CV 阶段最佳模型
best_cv_fold_idx = int(np.argmin(cv_val_losses))               # 0-based 索引
best_cv_model_path = fold_models[best_cv_fold_idx]
best_cv_epoch = fold_best_epochs[best_cv_fold_idx]

cv_best_save_path = os.path.join(Config.RESULT_SAVE_DIR, "best_model_cv.pth")
shutil.copy(best_cv_model_path, cv_best_save_path)              # 文件级复制
print(f"\n📥 CV best model (Fold {best_cv_fold_idx+1}, epoch {best_cv_epoch}) "
      f"copied to: {cv_best_save_path}")


# =============================================================================
# 阶段 4: 全量数据重新训练 (Refit on Full Data)
# =============================================================================
# 策略: 使用各折最佳 epoch 的平均值作为重训练轮数。
# 这是介于"欠训练"和"过拟合"之间的折中选择, 避免在完整数据上
# 使用固定的高 epoch 数导致过拟合。
print(f"\n{'='*60}")
print(" 🔁 Retraining on Full Dataset")
print(f"{'='*60}")

retrain_epochs = int(np.round(np.mean(fold_best_epochs)))
print(f"📈 Retrain Epochs = avg(best epochs) = {np.mean(fold_best_epochs):.1f} → {retrain_epochs}")

# ---- 准备全量数据 ----
# 将所有训练样本打包为 DataLoader (shuffle=True 打乱)
X_full = torch.tensor(x.reshape(-1, 1), device=device)
Y_full = torch.tensor(y.reshape(-1, 1), device=device)
full_loader = DataLoader(
    TensorDataset(X_full, Y_full),
    batch_size=Config.BATCH_SIZE,
    shuffle=True
)

# ---- 初始化最终模型 (从头训练) ----
final_model = Config.MODEL_CLASS().to(device)
criterion = Config.build_loss()
optimizer = Config.build_optimizer(final_model)
scheduler = Config.build_scheduler(optimizer)

# ---- 全量训练循环 ----
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

    # 每 50 轮或最后打印日志
    if epoch % 50 == 0 or epoch == retrain_epochs:
        print(f"  Epoch {epoch:4d}/{retrain_epochs} | train_loss = {epoch_loss:.6e}")

# ---- 保存最终模型到标准路径 ----
torch.save(final_model.state_dict(), Config.MODEL_SAVE_PATH)
print(f"\n✅ Final model (full-data retrain) saved to: {Config.MODEL_SAVE_PATH}")

# ---- 保存全量训练 loss 曲线 ----
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


# =============================================================================
# 阶段 5: 总结
# =============================================================================
print(f"\n{'='*60}")
print(" ✅ Training Pipeline Completed!")
print(f"{'='*60}")
print(f"• CV Mean Val Loss: {mean_val_loss:.6e} ± {std_val_loss:.6e}")
print(f"• Final Model (full retrain): {Config.MODEL_SAVE_PATH}")
print(f"• CV Best Model (backup):      {cv_best_save_path}")
print(f"• Results Directory:           {Config.RESULT_SAVE_DIR}")
print(f"{'='*60}")
