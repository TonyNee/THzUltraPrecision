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
# print(f'使用设备: {device}')
print(f'使用设备: gpu')

# ------------------------------
# 2. 加载数据
# ------------------------------
csv_path = './data/THz_train_20250928.csv'
df = pd.read_csv(csv_path)

x_data = df['Fexperiment_GHz'].values.astype(np.float32).reshape(-1, 1)
y_data = df['Fstandard_GHz'].values.astype(np.float32).reshape(-1, 1)

# ------------------------------
# 3. 划分训练集与验证集
# ------------------------------
X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train_np).to(device)
Y_train = torch.tensor(Y_train_np).to(device)
X_test = torch.tensor(X_test_np).to(device)
Y_test = torch.tensor(Y_test_np).to(device)

# ------------------------------
# 4. 定义模型
# ------------------------------
model = BPNN().to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.HuberLoss(delta=0.001) 
# optimizer = torch.optim.Rprop(model.parameters(), lr=0.035) 
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# ------------------------------
# 5. 模型训练
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
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.6f}')

# ------------------------------
# 6. 训练损失图
# ------------------------------
os.makedirs('./result', exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title('Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/loss_curve.png')
plt.close()

# ------------------------------
# 7. 验证集预测与可视化
# ------------------------------
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)

# 转回 CPU
Y_test_np = Y_test.cpu().numpy()
y_pred_np = y_pred_test.cpu().numpy()

plt.figure(figsize=(8, 5))
plt.plot(Y_test_np, label='Standard (True)')
plt.plot(y_pred_np, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Frequency (MHz)')
plt.title('Test Set: True vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./result/test_prediction_vs_truth.png')
plt.close()

# ------------------------------
# 8. 评价指标（训练集和验证集）
# ------------------------------
model.eval()
with torch.no_grad():
    # 训练集预测
    y_pred_train = model(X_train)
    
    # 训练集指标计算
    train_mse = criterion(y_pred_train, Y_train).item()
    train_r2 = 1 - (torch.sum((Y_train - y_pred_train) ** 2) / torch.sum((Y_train - torch.mean(Y_train)) ** 2)).item()
    
    # 计算训练集残差统计(MAE和标准差)
    train_residuals = (Y_train.cpu().numpy() - y_pred_train.cpu().numpy()) * 1000
    train_mae = np.mean(np.abs(train_residuals))
    train_std = np.std(train_residuals)
    
    # 验证集预测
    y_pred_test = model(X_test)
    
    # 验证集指标计算
    test_mse = criterion(y_pred_test, Y_test).item()
    test_r2 = 1 - (torch.sum((Y_test - y_pred_test) ** 2) / torch.sum((Y_test - torch.mean(Y_test)) ** 2)).item()
    
    # 计算验证集残差统计
    test_residuals = (Y_test.cpu().numpy() - y_pred_test.cpu().numpy()) * 1000
    test_mae = np.mean(np.abs(test_residuals))
    test_std = np.std(test_residuals)

# 以表格形式打印评估指标
print('\n===== 模型评价指标 =====')
metrics_table = pd.DataFrame({
    '指标': ['MAE', 'MSE', 'R²', 'S²'],
    '训练集': [train_mae, train_mse, train_r2, train_std],
    '验证集': [test_mae, test_mse, test_r2, test_std]
})
print(metrics_table.to_string(index=False, float_format='{:.6f}'.format))

# ------------------------------
# 9. 保存模型
# ------------------------------
torch.save(model.state_dict(), './model/bpnn_model_20250928.pth')
print('模型已保存至 ./model/bpnn_model_20250928.pth')

