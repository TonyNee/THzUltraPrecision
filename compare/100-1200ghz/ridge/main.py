"""
=============================================================================
模块名称: Ridge Regression 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用岭回归 (Ridge Regression) 作为基线模型, 与神经网络方法对比。
  岭回归在线性回归的基础上加入 L2 正则化项, 通过对系数大小的惩罚
  来防止过拟合, 提高模型泛化能力。

算法原理:
  目标函数: min ||y - Xw||^2 + α||w||^2
  α (alpha) 是正则化强度, 由 RidgeCV 通过交叉验证自动选择。
  使用 StandardScaler 进行标准化预处理, 确保正则化公平作用于各特征。

输出内容:
  - 最佳 alpha 值
  - 2x2 综合分析图
  - 系数路径图 (alpha vs coefficient)
  - 验证 MSE vs alpha 图
  - 频率残差对比图

使用方式:
  cd compare/0-40ghz/ridge && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置"""
    MODEL_TYPE = "RidgeRegression"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_csv(path):
    """
    读取两列 CSV 频率数据

    CSV 格式:
      第1行: 表头, 第1列: X (测量频率 GHz), 第2列: Y (真实频率 GHz)

    返回:
      X: (N, 1), y: (N, 1)
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
# 2. 标准化 + RidgeCV 训练
# =============================================================================
# 标准化: 将特征缩放为均值 0、标准差 1, 确保正则化对各特征公平
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)                    # 在训练集上拟合标准化参数
X_eval_scaled  = scaler.transform(X_eval)                         # 用相同参数标准化评估集

# alpha 搜索空间: 10^-6 ~ 10^4 对数均匀分布 200 个候选值
alphas = np.logspace(-6, 4, 200)

# RidgeCV: 内置留一交叉验证 (LOOCV) 自动选择最佳 alpha
ridgecv = RidgeCV(alphas=alphas)
ridgecv.fit(X_train_scaled, y_train.ravel())

print(f"Best alpha: {ridgecv.alpha_:.6e}")


# =============================================================================
# 3. 预测与残差
# =============================================================================
y_pred = ridgecv.predict(X_eval_scaled).reshape(-1, 1)

# 原始测量误差 vs 模型修正后残差
meas_residuals = X_eval - y_eval
pred_residuals = y_pred - y_eval

mse = mean_squared_error(y_eval, y_pred)
print(f"Eval MSE: {mse:.6f}")


# =============================================================================
# 图 1: 2x2 综合分析图
# =============================================================================
plt.figure(figsize=(12, 10))

# 子图 1: 散点 + 岭回归拟合线
plt.subplot(2, 2, 1)
plt.scatter(X_eval, y_eval, alpha=0.6, label="Actual")
plt.plot(X_eval, y_pred, linewidth=2, label="Ridge Fit")
plt.title("Scatter Plot with Ridge Regression (Eval)")
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

# 子图 4: 残差直方图
plt.subplot(2, 2, 4)
plt.hist(pred_residuals, bins=20, edgecolor="black", alpha=0.7)
plt.title("Residual Distribution")
plt.xlabel("Residuals (GHz)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "ridge_eval_overview.png"))
plt.close()


# =============================================================================
# 图 2: Ridge 系数路径 (展示正则化如何收缩系数)
# =============================================================================
ridge_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_.ravel())

ridge_coefs = np.array(ridge_coefs)

plt.figure(figsize=(8, 5))
plt.plot(alphas, ridge_coefs)
plt.xscale("log")                                                 # alpha 轴对数尺度
plt.xlabel("Alpha (正则化强度)")
plt.ylabel("Coefficient Value")
plt.title("Ridge Coefficient Path (α 越大系数越趋近于 0)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "ridge_coef_path.png"))
plt.close()


# =============================================================================
# 图 3: 验证 MSE vs Alpha (展示不同正则化强度下的预测性能)
# =============================================================================
cv_errors = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_cv_pred = ridge.predict(X_eval_scaled)
    cv_errors.append(mean_squared_error(y_eval, y_cv_pred))

plt.figure(figsize=(8, 5))
plt.plot(alphas, cv_errors)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("Validation MSE vs Alpha")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "ridge_mse_vs_alpha.png"))
plt.close()


# =============================================================================
# 图 4: 频率残差对比图 (MHz)
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
