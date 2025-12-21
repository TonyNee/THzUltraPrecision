import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

from model import BPNN

# ------------------------------
# 1. 设置设备
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: gpu')

# ------------------------------
# 2. 加载数据
# ------------------------------
csv_path = './data/THz_train_20250928.csv'
df = pd.read_csv(csv_path)

x_data = df['Fexperiment_GHz'].values.astype(np.float32).reshape(-1, 1)
y_data = df['Fstandard_GHz'].values.astype(np.float32).reshape(-1, 1)

# ------------------------------
# ⭐ 3. 输入输出归一化
# ------------------------------
x_mean, x_std = x_data.mean(), x_data.std()
y_mean, y_std = y_data.mean(), y_data.std()

x_norm = (x_data - x_mean) / x_std
y_norm = (y_data - y_mean) / y_std

np.savez("./model/norm_params_20250928.npz",
         x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

# ------------------------------
# 4. 划分训练集与验证集
# ------------------------------
X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(
    x_norm, y_norm, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train_np).to(device)
Y_train = torch.tensor(Y_train_np).to(device)
X_test = torch.tensor(X_test_np).to(device)
Y_test = torch.tensor(Y_test_np).to(device)

# ------------------------------
# 5. 定义模型 + HuberLoss + AdamW
# ------------------------------
model = BPNN().to(device)
criterion = nn.HuberLoss(delta=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# ------------------------------
# 6. 模型训练
# ------------------------------
epochs = 1000
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    output = model(X_train)
    loss = criterion(output, Y_train)

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.8f}')

# ------------------------------
# 7. 保存训练损失曲线
# ------------------------------
os.makedirs('./result', exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title('Training Loss (Huber)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/loss_curve.png')
plt.close()

# ------------------------------
# 8. 验证集预测（反归一化）
# ------------------------------
model.eval()
with torch.no_grad():
    y_pred_norm = model(X_test)

Y_test_GHz = Y_test.cpu().numpy() * y_std + y_mean
y_pred_GHz = y_pred_norm.cpu().numpy() * y_std + y_mean

# ------------------------------
# 9. 绘制验证集预测图
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(Y_test_GHz, label='Standard (True)')
plt.plot(y_pred_GHz, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Frequency (GHz)')
plt.title('Test Set: True vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/test_prediction_vs_truth.png')
plt.close()

# ------------------------------
# 10. 评价指标（GHz/MHz）
# ------------------------------
with torch.no_grad():
    y_pred_train_norm = model(X_train)
    y_pred_train_GHz = y_pred_train_norm.cpu().numpy() * y_std + y_mean
    Y_train_GHz = Y_train.cpu().numpy() * y_std + y_mean

    # ---- 训练集 ----
    train_residuals_MHz = (Y_train_GHz - y_pred_train_GHz) * 1000
    train_mae = np.mean(np.abs(train_residuals_MHz))
    train_std = np.std(train_residuals_MHz)
    train_mse = np.mean((Y_train_GHz - y_pred_train_GHz) ** 2)

    train_r2 = 1 - (
        np.sum((Y_train_GHz - y_pred_train_GHz)**2)
        / np.sum((Y_train_GHz - np.mean(Y_train_GHz))**2)
    )

    # ---- 验证集 ----
    test_residuals_MHz = (Y_test_GHz - y_pred_GHz) * 1000
    test_mae = np.mean(np.abs(test_residuals_MHz))
    test_std = np.std(test_residuals_MHz)
    test_mse = np.mean((Y_test_GHz - y_pred_GHz)**2)

    test_r2 = 1 - (
        np.sum((Y_test_GHz - y_pred_GHz)**2)
        / np.sum((Y_test_GHz - np.mean(Y_test_GHz))**2)
    )

# 输出表格
print('\n===== 模型评价指标（GHz / MHz） =====')
metrics_table = pd.DataFrame({
    '指标': ['MAE (MHz)', 'MSE (GHz^2)', 'R²', 'Std (MHz)'],
    '训练集': [train_mae, train_mse, train_r2, train_std],
    '验证集': [test_mae, test_mse, test_r2, test_std]
})
print(metrics_table.to_string(index=False, float_format='{:.6f}'.format))

# ------------------------------
# 11. 保存模型
# ------------------------------
torch.save(model.state_dict(), './model/bpnn_model_20250928.pth')
print('模型已保存至 ./model/bpnn_model_20250928.pth')
