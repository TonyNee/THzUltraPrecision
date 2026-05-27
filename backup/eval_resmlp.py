"""
=============================================================================
模块名称: backup/eval_resmlp.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (早期备份版本)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  早期简化版 ResMLP 评估脚本。
  加载模型进行推理, 计算 MAE/STD/MSE, 保存预测 CSV 和残差图。

与当前 eval.py 的主要差异:
  - 无分位数误差指标 (E1s/E2s/E3s 等)
  - 无 R² 决定系数
  - 无 747 GHz 通道分界标注
  - 无残差概率密度分布 (PDF) 图
  - 无分通道 MAE 计算
  - 硬编码路径 (不使用 Config 类)

注意: 此文件已废弃, 当前项目使用 eval.py 进行评估。
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from model_resmlp import HighPrecCalibrator                    # 早期独立模型

# =============================================================================
# 配置 (硬编码)
# =============================================================================
EVAL_CSV = '../data/20250928/THz_eval_20250928.csv'             # 评估集路径
MODEL_PATH = './model/resmlp_calibration_best.pth'              # 模型权重路径
SAVE_DIR = "./result_resmlp/"                                    # 结果输出目录

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# 1. 加载评估数据
# =============================================================================
df = pd.read_csv(EVAL_CSV)
x_eval = df.iloc[:, 0].values.astype(np.float32)                # 测量频率 (GHz)
y_eval = df.iloc[:, 1].values.astype(np.float32)                # 真实频率 (GHz)

X = torch.tensor(x_eval.reshape(-1, 1)).to(device)              # (N, 1) 输入张量
Y = torch.tensor(y_eval.reshape(-1, 1)).to(device)              # (N, 1) 目标张量

# =============================================================================
# 2. 加载训练好的模型
# =============================================================================
model = HighPrecCalibrator().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()                                                     # 切换到推理模式

# =============================================================================
# 3. 推理预测
# =============================================================================
with torch.no_grad():                                            # 不计算梯度
    y_pred = model(X).cpu().numpy()

y_true = Y.cpu().numpy()

# =============================================================================
# 4. 误差指标计算
# =============================================================================
# 残差 = 真实值 - 预测值 (单位: GHz)
residuals = (y_true - y_pred).flatten()

# 指标计算: 1 GHz = 1000 MHz
MAE = np.mean(np.abs(residuals)) * 1000                          # 平均绝对误差 (MHz)
STD = np.std(residuals) * 1000                                   # 残差标准差 (MHz)
MSE = np.mean(residuals ** 2)                                    # 均方误差 (GHz^2)

print("\n===== ResMLP 评估结果 =====")
print(f"MAE: {MAE:.3f} MHz")
print(f"STD: {STD:.3f} MHz")
print(f"MSE: {MSE:.10f} GHz^2")

# =============================================================================
# 5. 保存预测结果 CSV
# =============================================================================
df_out = pd.DataFrame({
    "F_test_GHz": x_eval,                                       # 测试频率
    "F_true_GHz": y_eval,                                       # 真实频率
    "F_pred_GHz": y_pred.flatten(),                             # 模型预测频率
    "Residual(GHz)": residuals                                  # 残差 (GHz)
})
df_out.to_csv(os.path.join(SAVE_DIR, "resmlp_predictions.csv"), index=False)

# =============================================================================
# 6. 绘制残差图 (按样本序号)
# =============================================================================
plt.figure(figsize=(8, 5))
plt.plot(residuals * 1000)                                      # 转换为 MHz
plt.axhline(0, linestyle="--", color="black")                   # 零残差参考线
plt.xlabel("Sample Index")
plt.ylabel("Residual (MHz)")
plt.title("ResMLP Frequency Calibration Residuals")
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "residual_plot.png"))
plt.close()
