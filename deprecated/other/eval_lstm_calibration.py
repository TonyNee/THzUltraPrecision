"""
=============================================================================
模块名称: deprecated/other/eval_lstm_calibration.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (LSTM 实验, 已废弃)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  LSTM 频率校准网络的评估脚本。
  加载训练时保存的归一化参数和最佳模型, 对评估集进行推理,
  反归一化后计算 MAE/STD/MSE。

评估流程:
  1. 加载评估数据, 计算残差
  2. 加载归一化参数 (.npz)
  3. 构造滑动窗口序列, 归一化
  4. 加载模型, 推理, 反归一化
  5. 取序列最后一个时间步的预测, 计算指标

注意: 此文件已废弃。
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
from lstm_model import LSTMCalibNet

# =============================================================================
# 配置
# =============================================================================
SEQ_LEN = 16                                                     # 序列长度 (需与训练时一致)
EVAL_CSV = "./data/THz_eval_20250928.csv"

# =============================================================================
# 1. 加载评估数据
# =============================================================================
df = pd.read_csv(EVAL_CSV)
x = df["Fexperiment_GHz"].values.astype(np.float32)              # 测量频率 (GHz)
y = df["Fstandard_GHz"].values.astype(np.float32)                # 标准频率 (GHz)

# =============================================================================
# 2. 加载训练时保存的归一化参数
# =============================================================================
norm = np.load("./model/lstm_norm_params.npz")
x_mean, x_std = norm["x_mean"], norm["x_std"]                    # 频率归一化参数
r_mean, r_std = norm["r_mean"], norm["r_std"]                    # 残差归一化参数

# =============================================================================
# 3. 构造序列 (滑动窗口)
# =============================================================================
def build_seq(arr, seq_len):
    """
    将一维数组转换为重叠的滑动窗口序列

    返回:
      np.array: (N - seq_len, seq_len)
    """
    Xs = []
    for i in range(len(arr) - seq_len):
        Xs.append(arr[i:i+seq_len])
    return np.array(Xs)

res = y - x                                                      # 真实残差 (GHz)
X_seq = build_seq(x, SEQ_LEN)                                    # 频率序列
R_true = build_seq(res, SEQ_LEN)                                 # 真实残差序列

# ---- 归一化 (仅对输入 X, 使用训练时的均值和标准差) ----
Xn = (X_seq - x_mean) / (x_std + 1e-12)

# =============================================================================
# 4. 加载模型
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMCalibNet()
model.load_state_dict(torch.load("./model/lstm_calibration_best.pth", map_location=device))
model.to(device)
model.eval()

# =============================================================================
# 5. 推理 & 反归一化
# =============================================================================
with torch.no_grad():
    # 转为张量: (N, seq_len) -> (N, seq_len, 1)
    pred_norm = model(torch.tensor(Xn).float().unsqueeze(-1).to(device)).cpu().numpy()

# 反归一化: 从 z-score 还原到 GHz
pred_res = pred_norm * r_std + r_mean

# =============================================================================
# 6. 评估: 取序列最后一个时间步的预测
# =============================================================================
# LSTM 预测整个序列, 但实际只关心最后一个时间步的修正量
residuals = R_true[:, -1] - pred_res[:, -1]

# 误差指标
mae = np.mean(np.abs(residuals)) * 1000                           # MHz
std = np.std(residuals) * 1000                                    # MHz (标准差)
mse = np.mean((residuals)**2)                                     # GHz^2

print("\n===== LSTM 评估结果 =====")
print(f"MAE: {mae:.6f} MHz")
print(f"STD: {std:.6f} MHz")
print(f"MSE: {mse:.10f} GHz^2")
