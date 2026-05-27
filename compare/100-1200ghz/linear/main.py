"""
=============================================================================
模块名称: Linear Regression 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用最基础的线性回归 (Ordinary Least Squares) 作为所有非线性模型的比较基线。
  如果非线性模型的指标明显优于线性回归, 则验证了非线性建模的必要性;
  如果差异很小, 说明频率校准问题本身近似线性关系。

算法原理:
  拟合一条直线 y = wx + b 来最小化残差平方和。
  作为最简模型, 它提供了可解释的基准, 所有复杂模型的增益都以此为参照。

使用方式:
  cd compare/0-40ghz/linear && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置 (模型类型 + 输出目录)"""
    MODEL_TYPE = "LinearRegression"                               # 模型标识
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))  # 结果保存到脚本所在目录


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_csv(path):
    """
    读取两列 CSV 频率数据

    CSV 格式:
      第1行: 表头
      第1列: 频率测量值 X (GHz)
      第2列: 频率真实值 Y (GHz)

    返回:
      X: (N, 1), y: (N, 1)
    """
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,                                               # 跳过表头行
        usecols=(0, 1)
    )
    X = data[:, 0:1]
    y = data[:, 1:2]
    return X, y


X_train, y_train = load_csv("./input/20251216/train.csv")
X_eval, y_eval   = load_csv("./input/20251216/eval.csv")


# =============================================================================
# 2. 训练线性回归模型 (最小二乘法 OLS)
# =============================================================================
# 线性回归没有超参数, 直接拟合即可
model = LinearRegression()
model.fit(X_train, y_train)


# =============================================================================
# 3. 预测与残差评估
# =============================================================================
y_pred = model.predict(X_eval)

# 残差定义
# meas_residuals: 原始测量偏差 (校准前)
# pred_residuals: 模型修正后残差 (校准后)
meas_residuals = X_eval - y_eval
pred_residuals = y_pred - y_eval

# 误差指标
mse = mean_squared_error(y_pred, y_eval)
mae = np.mean(np.abs(pred_residuals)) * 1000                      # MHz
rmse = np.sqrt(mse) * 1000                                        # MHz

print(f"Eval MAE: {mae:.4f} MHz")
print(f"Eval MSE: {mse:.6f} GHz^2")
print(f"Eval RMSE: {rmse:.4f} MHz")

maxae = np.max(np.abs(pred_residuals)) * 1000
print(f"Eval MaxAE: {maxae:.4f} MHz")


# =============================================================================
# 图 1: 2x2 综合分析图
# =============================================================================
plt.figure(figsize=(12, 10))

# ---- 子图 1: 散点图 + 回归线 ----
plt.subplot(2, 2, 1)
plt.scatter(X_eval, y_eval, alpha=0.6, label="Actual Data")
plt.plot(X_eval, y_pred, linewidth=2, label="Regression Line")    # 拟合直线
plt.title("Scatter Plot with Regression Line (Eval)")
plt.xlabel("X (Measured Frequency, GHz)")
plt.ylabel("Y (True Frequency, GHz)")
plt.legend()

# ---- 子图 2: 残差图 (残差 vs X) ----
plt.subplot(2, 2, 2)
plt.scatter(X_eval, pred_residuals, alpha=0.6)
plt.hlines(0, X_eval.min(), X_eval.max(), linestyles="--")        # 零残差线
plt.title("Residual Plot (Eval)")
plt.xlabel("X (GHz)")
plt.ylabel("Residuals (GHz)")

# ---- 子图 3: 预测值 vs 实际值 ----
plt.subplot(2, 2, 3)
plt.scatter(y_eval, y_pred, alpha=0.6)
min_y, max_y = y_eval.min(), y_eval.max()
plt.plot([min_y, max_y], [min_y, max_y], linestyle="--", label="Perfect Fit")  # 完美预测线
plt.title("Predicted vs Actual (Eval)")
plt.xlabel("Actual Y (GHz)")
plt.ylabel("Predicted Y (GHz)")
plt.legend()

# ---- 子图 4: 残差分布直方图 ----
plt.subplot(2, 2, 4)
plt.hist(pred_residuals, bins=20, edgecolor="black", alpha=0.7)
plt.title("Residuals Distribution (Eval)")
plt.xlabel("Residuals (GHz)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "linear_eval_overview.png"))
plt.close()


# =============================================================================
# 图 2: 频率残差对比图 (MHz)
# =============================================================================
x_freq = X_eval.squeeze()
idx = np.argsort(x_freq)                                          # 按频率升序

save_name = f"residual_plot_{Config.MODEL_TYPE}.png"
title_str = f"{Config.MODEL_TYPE} Frequency Residuals"

plt.figure(figsize=(8, 5))

# 测量误差 vs 模型修正误差 (MHz)
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

# 左上角显示 MAE
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
