import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from model import BPNN
# ------------------------------
# 1. 设置设备（CUDA or CPU）
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'使用设备: {device}')
print(f'使用设备: gpu')

# ------------------------------
# 2. 加载测试数据
# ------------------------------
test_csv_path = './data/THz_eval_20250928.csv' 
df_test = pd.read_csv(test_csv_path)

x_test = df_test.iloc[:, 0].values.astype(np.float32)
y_test = df_test.iloc[:, 1].values.astype(np.float32)

X_test = torch.tensor(x_test.reshape(-1, 1)).to(device)
Y_test = torch.tensor(y_test.reshape(-1, 1)).to(device)

# ------------------------------
# 4. 加载模型权重
# ------------------------------
model = BPNN().to(device)
model_path = './model/bpnn_model_20250928.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f'成功加载模型: {model_path}')

# ------------------------------
# 5. 模型预测
# ------------------------------
with torch.no_grad():
    y_pred_test = model(X_test)

# ------------------------------
# 6. 可视化结果
# ------------------------------
y_true = Y_test.cpu().numpy()
y_pred = y_pred_test.cpu().numpy()

plt.figure(figsize=(8, 5))
plt.plot(y_true, label='Standard (True)')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Frequency (GHz)')
plt.title('Test Set: True vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_dir = './result/'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'test_prediction_vs_truth.png'))
plt.close()

# ------------------------------
# 7. 计算评价指标
# ------------------------------
criterion = nn.MSELoss()
mse = criterion(y_pred_test, Y_test).item()
r2 = 1 - (torch.sum((Y_test - y_pred_test) ** 2) / torch.sum((Y_test - torch.mean(Y_test)) ** 2)).item()
mae = torch.mean(torch.abs(y_pred_test - Y_test)).item()

print(f'\n测试集 MAE: {mae:.6f}')
print(f'测试集 MSE: {mse:.6f}')
print(f'测试集 R²: {r2:.6f}')

# ------------------------------
# 8. 保存预测结果 
# ------------------------------
save_pred_path = os.path.join(output_dir, 'test_predictions.csv')
df_pred = pd.DataFrame({
    'Fexperiment_GHz': x_test,
    'Fstandard_GHz': y_test,
    'Fpredicted_GHz': y_pred.flatten()
})
df_pred.to_csv(save_pred_path, index=False)
print(f'预测结果已保存至 {save_pred_path}')

# ------------------------------
# 9. 绘制残差曲线
# ------------------------------
residuals = (y_true - y_pred)*1000  # 残差 = 真值 - 预测值
# 输出残差的最大值和最小值
print(f"残差最大值: {np.max(abs(residuals)):.12f} MHz")
print(f"残差最小值: {np.min(abs(residuals)):.12f} MHz")

plt.figure(figsize=(8, 5))
plt.plot(residuals, label='Residuals', color='orange')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Sample Index')
plt.ylabel('Residual (MHz)')
plt.title('Residual Plot (Standard - Predicted)')
plt.legend()
plt.grid(True)

# plt.ylim(-4, 10)                         # 范围 [-10, 4]
# plt.yticks(np.arange(-4, 10.1, 2))       # 每 2 GHz 一刻度

plt.tight_layout()

residual_plot_path = os.path.join(output_dir, 'residual_plot.png')
plt.savefig(residual_plot_path)
plt.close()
print(f'残差图已保存至 {residual_plot_path}')

