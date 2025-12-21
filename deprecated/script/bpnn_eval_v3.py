import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from model import BPNN

# ------------------------------
# 1. 设置设备
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: gpu')

# ------------------------------
# 2. 加载测试数据
# ------------------------------
test_csv_path = './data/THz_eval_20250928.csv'
df_test = pd.read_csv(test_csv_path)

x_test_raw = df_test.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)
y_test_raw = df_test.iloc[:, 1].values.astype(np.float32).reshape(-1, 1)

# ------------------------------
# ⭐ 3. 加载训练时保存的归一化参数
# ------------------------------
# 建议你在训练脚本中保存如下参数：x_mean, x_std, y_mean, y_std
norm_param_path = "./model/norm_params_20250928.npz"
norm = np.load(norm_param_path)

x_mean = norm["x_mean"]
x_std = norm["x_std"]
y_mean = norm["y_mean"]
y_std = norm["y_std"]

print("成功加载归一化参数。")

# ------------------------------
# ⭐ 4. 对测试数据做相同归一化（必须与训练一致）
# ------------------------------
x_test_norm = (x_test_raw - x_mean) / x_std

X_test = torch.tensor(x_test_norm.astype(np.float32)).to(device)
Y_test_raw = y_test_raw  # 保留原始 GHz 方便后面对比

# ------------------------------
# 5. 加载模型
# ------------------------------
model = BPNN().to(device)
model_path = './model/bpnn_model_20250928.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f'成功加载模型: {model_path}')

# ------------------------------
# ⭐ 6. 做预测（输出为归一化后的 y_norm）
# ------------------------------
with torch.no_grad():
    y_pred_norm = model(X_test)

# ------------------------------
# ⭐ 7. 反归一化（恢复到 GHz）
# ------------------------------
y_pred = y_pred_norm.cpu().numpy() * y_std + y_mean
y_true = Y_test_raw

# ------------------------------
# 8. 绘制预测图
# ------------------------------
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
# ⭐ 9. 使用反归一化后的值计算评价指标
# ------------------------------
mae = np.mean(np.abs(y_pred - y_true))
mse = np.mean((y_pred - y_true) ** 2)
r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

print(f'\n测试集 MAE: {mae*1000:.6f} MHz')
print(f'测试集 MSE: {mse:.12f} (GHz^2)')
print(f'测试集 R²: {r2:.6f}')

# ------------------------------
# 10. 保存预测结果
# ------------------------------
save_pred_path = os.path.join(output_dir, 'test_predictions.csv')
df_pred = pd.DataFrame({
    'Fexperiment_GHz': x_test_raw.flatten(),
    'Fstandard_GHz': y_true.flatten(),
    'Fpredicted_GHz': y_pred.flatten()
})
df_pred.to_csv(save_pred_path, index=False)
print(f'预测结果已保存至 {save_pred_path}')

# ------------------------------
# ⭐ 11. 绘制残差曲线（MHz）
# ------------------------------
residuals = (y_true - y_pred) * 1000

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
plt.tight_layout()

residual_plot_path = os.path.join(output_dir, 'residual_plot.png')
plt.savefig(residual_plot_path)
plt.close()
print(f'残差图已保存至 {residual_plot_path}')
