import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# ------------------------------
# 1. 设置设备（CUDA or CPU）
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: gpu')

# ------------------------------
# 2. 加载测试数据
# ------------------------------
test_csv_path = './data/eval_v3.csv'
df_test = pd.read_csv(test_csv_path)

# 前 N 列是实验值，最后一列是标准值
x_test_all = df_test.iloc[:, :-1].values.astype(np.float32)  # 实验值矩阵 (N行 × M列)
y_test = df_test.iloc[:, -1].values.astype(np.float32)       # 标准值 (N行, )

X_test_all = torch.tensor(x_test_all).to(device)
Y_test = torch.tensor(y_test.reshape(-1, 1)).to(device)

# ------------------------------
# 3. 定义模型结构
# ------------------------------
class BPNN(nn.Module):
    def __init__(self):
        super(BPNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    def forward(self, x):
        return self.model(x)

# ------------------------------
# 4. 加载模型权重
# ------------------------------
model = BPNN().to(device)
model_path = './model/bpnn_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f'成功加载模型: {model_path}')

# ------------------------------
# 5. 遍历每一列实验值，预测并计算MAE
# ------------------------------
mae_list = []
residuals_dict = {}

with torch.no_grad():
    for col_idx in range(x_test_all.shape[1]):
        x_col = x_test_all[:, col_idx].reshape(-1, 1)
        X_col = torch.tensor(x_col).to(device)

        y_pred_col = model(X_col).cpu().numpy().flatten()
        y_true = y_test

        # 计算 MAE
        mae = np.mean(np.abs(y_pred_col - y_true))
        mae_list.append(mae)

        # 保存残差曲线
        residuals_dict[f'exp_col_{col_idx+1}'] = y_true - y_pred_col

# ------------------------------
# 6. 输出 MAE统计
# ------------------------------
mae_array = np.array(mae_list)
print("\n===== MAE统计结果 =====")
print(f"平均 MAE: {np.mean(mae_array):.6f}")
print(f"最大 MAE: {np.max(mae_array):.6f}")
print(f"最小 MAE: {np.min(mae_array):.6f}")

# ------------------------------
# 7. 残差图绘制（多条曲线）
# ------------------------------
output_dir = './result/'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
for idx, (label, residuals) in enumerate(residuals_dict.items()):
    plt.plot(residuals, label=label)

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Sample Index')
plt.ylabel('Residual (GHz)')
plt.title('Residual Plot (Standard - Predicted)')
# plt.legend()
plt.grid(True)

plt.ylim(-10, 4)                         # 范围 [-10, 4]
plt.yticks(np.arange(-10, 4.1, 2))       # 每 2 GHz 一刻度

plt.tight_layout()

residual_plot_path = os.path.join(output_dir, 'residual_plot_multicol.png')
plt.savefig(residual_plot_path)
plt.close()
print(f'残差图已保存至 {residual_plot_path}')
