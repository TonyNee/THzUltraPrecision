"""
=============================================================================
模块名称: ElasticNet 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用弹性网络回归 (ElasticNet) 作为基线模型。
  弹性网络结合了 L1 (Lasso) 和 L2 (Ridge) 正则化的优点:
  既能产生稀疏解 (L1), 又能处理特征间的相关性 (L2)。

算法原理:
  目标函数: min ||y - Xw||^2 + α * [ρ||w||1 + (1-ρ)/2 * ||w||^2]
  其中 α 控制总体正则化强度, ρ (l1_ratio) 控制 L1/L2 的混合比例。
  ρ=1 退化为 Lasso, ρ=0 退化为 Ridge。

超参数: alpha=1e-3 (正则化强度), l1_ratio=0.7 (偏向 L1 的混合比例)

使用方式:
  cd compare/0-40ghz/elastic && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置"""
    MODEL_TYPE = "ElasticNet"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_csv(path):
    """
    读取两列 CSV 频率数据

    CSV 格式:
      第1行: 表头, 第1列: X (测量频率 GHz), 第2列: Y (真实频率 GHz)
    """
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1)
    )
    X = data[:, 0:1]
    y = data[:, 1:2]
    return X, y


X_train, y_train = load_csv("./input/20251216/train.csv")
X_eval,  y_eval  = load_csv("./input/20251216/eval.csv")


# =============================================================================
# 2. 标准化 + ElasticNet 训练
# =============================================================================
# 标准化预处理: 弹性网络的正则化项对特征尺度敏感
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled  = scaler.transform(X_eval)

# ElasticNet: alpha 控制正则化强度, l1_ratio 控制 L1/L2 混合比
# l1_ratio=0.7 表示 70% L1 + 30% L2, 偏向稀疏解
elastic = ElasticNet(
    alpha=1e-3,                                                   # 正则化强度
    l1_ratio=0.7,                                                 # L1/(L1+L2) 比例
    max_iter=10000,                                               # 最大迭代次数
    random_state=42
)
elastic.fit(X_train_scaled, y_train.ravel())


# =============================================================================
# 3. 预测与残差
# =============================================================================
y_pred = elastic.predict(X_eval_scaled).reshape(-1, 1)

meas_residuals = X_eval - y_eval                                  # 校准前
pred_residuals = y_pred - y_eval                                  # 校准后

mse = mean_squared_error(y_eval, y_pred)
print(f"Eval MSE: {mse:.6f}")


# =============================================================================
# 图 1: 2x2 综合分析图
# =============================================================================
plt.figure(figsize=(12, 10))

# 子图 1: 散点 + 弹性网络拟合线
plt.subplot(2, 2, 1)
plt.scatter(X_eval, y_eval, alpha=0.6, label="Actual")
plt.plot(X_eval, y_pred, linewidth=2, label="ElasticNet Fit")
plt.title("Scatter Plot with ElasticNet (Eval)")
plt.xlabel("X (GHz)")
plt.ylabel("Y (GHz)")
plt.legend()

# 子图 2: 残差图
plt.subplot(2, 2, 2)
plt.scatter(X_eval, pred_residuals, alpha=0.6)
plt.hlines(0, X_eval.min(), X_eval.max(), linestyles="--")
plt.title("Residual Plot (Eval)")
plt.xlabel("X (GHz)")
plt.ylabel("Residuals (GHz)")

# 子图 3: 预测 vs 实际
plt.subplot(2, 2, 3)
plt.scatter(y_eval, y_pred, alpha=0.6)
min_y, max_y = y_eval.min(), y_eval.max()
plt.plot([min_y, max_y], [min_y, max_y], linestyle="--", label="Perfect Fit")
plt.title("Predicted vs Actual")
plt.xlabel("Actual Y (GHz)")
plt.ylabel("Predicted Y (GHz)")
plt.legend()

# 子图 4: 残差分布直方图
plt.subplot(2, 2, 4)
plt.hist(pred_residuals, bins=20, edgecolor="black", alpha=0.7)
plt.title("Residual Distribution")
plt.xlabel("Residuals (GHz)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "elasticnet_eval_overview.png"))
plt.close()


# =============================================================================
# 图 2: ElasticNet 系数 (展示 L1+L2 混合约束的效果)
# =============================================================================
plt.figure(figsize=(6, 4))
plt.bar([0], elastic.coef_, color="steelblue")
plt.axhline(0, linestyle="--", color="black")
plt.title("ElasticNet Coefficient (L1 + L2 混合约束)")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "elasticnet_coefficients.png"))
plt.close()


# =============================================================================
# 图 3: 频率残差对比图 (MHz)
# =============================================================================
x_freq = X_eval.squeeze()
idx = np.argsort(x_freq)

save_name = f"residual_plot_{Config.MODEL_TYPE}.png"
title_str = f"{Config.MODEL_TYPE} Frequency Residuals"

plt.figure(figsize=(8, 5))
plt.plot(
    x_freq[idx],
    meas_residuals.squeeze()[idx] * 1000,
    label="MEASURED ERROR"
)
plt.plot(
    x_freq[idx],
    pred_residuals.squeeze()[idx] * 1000,
    label="PREDICTED ERROR"
)

plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residuals (MHz)")
plt.title(title_str)
plt.grid(True)

meas_mae = np.mean(np.abs(meas_residuals)) * 1000
pred_mae = np.mean(np.abs(pred_residuals)) * 1000

plt.text(
    0.05, 0.95,
    f"MEAS_MAE: {meas_mae:.4f} MHz\nPRED_MAE: {pred_mae:.4f} MHz",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, save_name))
plt.close()
