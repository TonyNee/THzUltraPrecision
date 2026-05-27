"""
=============================================================================
模块名称: Lasso Regression 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用 Lasso 回归作为基线模型。
  Lasso (Least Absolute Shrinkage and Selection Operator) 在线性回归
  的基础上加入 L1 正则化, 能够将不重要的特征系数压缩至 0,
  实现自动特征选择。

算法原理:
  目标函数: min ||y - Xw||^2 + α||w||1
  L1 惩罚项使得部分系数精确为 0 (稀疏性), 而 Ridge 的 L2 只能趋近于 0。
  在本任务的单特征场景中, Lasso 主要体现为对系数的收缩效果。

超参数: alpha=1e-3, 使用 StandardScaler 标准化预处理

使用方式:
  cd compare/0-40ghz/lasso && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置"""
    MODEL_TYPE = "LassoRegression"
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
    X = data[:, 0:1]                                              # (N, 1)
    y = data[:, 1:2]                                              # (N, 1)
    return X, y


X_train, y_train = load_csv("./input/20251216/train.csv")
X_eval,  y_eval  = load_csv("./input/20251216/eval.csv")


# =============================================================================
# 2. 标准化 + Lasso 训练
# =============================================================================
# 标准化: Lasso 的 L1 惩罚对特征尺度敏感, 必须标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled  = scaler.transform(X_eval)

alpha = 1e-3                                                      # L1 正则化强度
lasso = Lasso(alpha=alpha, max_iter=10000)
lasso.fit(X_train_scaled, y_train.ravel())


# =============================================================================
# 3. 预测与残差
# =============================================================================
y_pred = lasso.predict(X_eval_scaled).reshape(-1, 1)

meas_residuals = X_eval - y_eval                                  # 校准前
pred_residuals = y_pred - y_eval                                  # 校准后

mse = mean_squared_error(y_eval, y_pred)
print(f"Eval MSE: {mse:.6f}")


# =============================================================================
# 图 1: 2x2 综合分析图
# =============================================================================
plt.figure(figsize=(12, 10))

# 子图 1: 散点 + Lasso 回归线
plt.subplot(2, 2, 1)
plt.scatter(X_eval, y_eval, alpha=0.6, label="Actual")
plt.plot(X_eval, y_pred, linewidth=2, label="Lasso Fit")
plt.title("Scatter Plot with Lasso Regression (Eval)")
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
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "lasso_eval_overview.png"))
plt.close()


# =============================================================================
# 图 2: Lasso 系数 (展示 L1 稀疏性的效果)
# =============================================================================
plt.figure(figsize=(6, 4))
plt.bar([0], lasso.coef_, color="cyan")
plt.axhline(0, linestyle="--", color="black")
plt.title("Lasso Coefficient (L1 稀疏性: 不重要特征系数被压缩至 0)")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "lasso_coefficients.png"))
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
