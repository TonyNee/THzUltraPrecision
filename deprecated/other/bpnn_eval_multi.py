"""
=============================================================================
模块名称: deprecated/other/bpnn_eval_multi.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (多列实验, 已废弃)
作　　者: TonyNee
创建日期: 2025-10
最后修改: 2026-05-27
=============================================================================

功能概述:
  多列实验值的 BPNN 评估脚本。
  处理包含多个实验频率列 (同一物理量多次测量) 的 CSV 数据,
  对每一列分别预测并计算 MAE, 最后绘制所有列的残差曲线汇总图。

应用场景: 同一频率点多次重复测量的质量评估

评估流程:
  1. 加载多列测试数据 (前 N-1 列为实验值, 最后一列为标准值)
  2. 加载训练好的 BPNN 模型
  3. 逐列预测并计算 MAE
  4. 统计 MAE 分布 (平均/最大/最小)
  5. 绘制所有列的残差曲线 (同一张图叠加)

注意: 此文件已废弃, 当前项目使用更简洁的 eval.py。
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. 设备配置
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {"GPU" if device.type == "cuda" else "CPU"}')

# =============================================================================
# 2. 加载多列测试数据
# =============================================================================
# CSV 格式: 前 N-1 列是实验测量频率, 最后一列是标准真实频率
test_csv_path = './data/eval_v3.csv'
df_test = pd.read_csv(test_csv_path)

x_test_all = df_test.iloc[:, :-1].values.astype(np.float32)      # 实验值矩阵 (N行 x M列)
y_test = df_test.iloc[:, -1].values.astype(np.float32)           # 标准值 (N行,)

X_test_all = torch.tensor(x_test_all).to(device)
Y_test = torch.tensor(y_test.reshape(-1, 1)).to(device)

# =============================================================================
# 3. 定义 BPNN 模型结构 (需与训练时一致)
# =============================================================================
class BPNN(nn.Module):
    """
    简单 BPNN: 2 层全连接 (1 -> 20 -> 1), ReLU 激活

    注意: 内联定义是为了确保结构的独立性, 不依赖外部 model.py
    """
    def __init__(self):
        super(BPNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 20),                                    # 输入层 -> 隐藏层
            nn.ReLU(),                                           # ReLU 非线性激活
            nn.Linear(20, 1)                                     # 隐藏层 -> 输出层
        )

    def forward(self, x):
        """前向传播: 输入 (batch, 1) -> 输出 (batch, 1)"""
        return self.model(x)

# =============================================================================
# 4. 加载预训练模型权重
# =============================================================================
model = BPNN().to(device)
model_path = './model/bpnn_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f'成功加载模型: {model_path}')

# =============================================================================
# 5. 逐列预测并计算 MAE
# =============================================================================
mae_list = []                                                    # 各列的 MAE
residuals_dict = {}                                              # 各列的残差曲线

with torch.no_grad():
    for col_idx in range(x_test_all.shape[1]):
        # 取当前列 (N,) -> (N, 1)
        x_col = x_test_all[:, col_idx].reshape(-1, 1)
        X_col = torch.tensor(x_col).to(device)

        y_pred_col = model(X_col).cpu().numpy().flatten()        # 预测值
        y_true = y_test

        # 计算 MAE (平均绝对误差)
        mae = np.mean(np.abs(y_pred_col - y_true))
        mae_list.append(mae)

        # 保存残差 (真实值 - 预测值)
        residuals_dict[f'exp_col_{col_idx+1}'] = y_true - y_pred_col

# =============================================================================
# 6. 输出 MAE 统计
# =============================================================================
mae_array = np.array(mae_list)
print("\n===== MAE统计结果 =====")
print(f"平均 MAE: {np.mean(mae_array):.6f}")
print(f"最大 MAE: {np.max(mae_array):.6f}")                     # 最差列
print(f"最小 MAE: {np.min(mae_array):.6f}")                     # 最佳列

# =============================================================================
# 7. 残差图绘制 (所有列叠加)
# =============================================================================
output_dir = './result/'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))

# 每列绘制一条残差曲线
for idx, (label, residuals) in enumerate(residuals_dict.items()):
    plt.plot(residuals, label=label)

plt.axhline(0, color='black', linestyle='--', linewidth=1)       # 零残差参考线
plt.xlabel('Sample Index')
plt.ylabel('Residual (GHz)')
plt.title('Residual Plot (Standard - Predicted) for All Columns')

plt.ylim(-10, 4)                                                 # 固定 Y 轴范围
plt.yticks(np.arange(-10, 4.1, 2))                              # 每 2 GHz 一个刻度
plt.grid(True)

plt.tight_layout()

residual_plot_path = os.path.join(output_dir, 'residual_plot_multicol.png')
plt.savefig(residual_plot_path)
plt.close()
print(f'残差图已保存至 {residual_plot_path}')
