"""
=============================================================================
模块名称: Polynomial Regression 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用多项式回归 (Polynomial Regression) 作为基线模型。
  通过构造原始特征的高次幂作为新特征, 再用线性回归拟合,
  从而实现对非线性关系的建模。

算法原理:
  1. 用 PolynomialFeatures 将输入 x 扩展为 [1, x, x^2, ..., x^d]
  2. 对扩展后的特征用普通线性回归 (OLS) 拟合
  3. 阶数 d 越高, 模型越灵活, 但过高会过拟合

实验设计:
  测试阶数 d = 1~10, 绘制所有拟合曲线、残差、MSE vs 阶数,
  自动选择 MSE 最小的阶数作为最优模型。

使用方式:
  cd compare/0-40ghz/poly && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置"""
    MODEL_TYPE = "PolynomialRegression"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_csv(path):
    """
    读取两列 CSV 频率数据: 第1列 X (测量频率 GHz), 第2列 Y (真实频率 GHz)
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
# 2. 多项式阶数候选列表
# =============================================================================
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                       # 测试 1~10 阶多项式


# =============================================================================
# 图 1: 不同阶数拟合 + 残差 + MSE vs 阶数
# =============================================================================
plt.figure(figsize=(12, 10))

mse_list = []                                                     # 记录各阶数的评估 MSE

for degree in degrees:
    # 构造多项式特征
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)                    # 训练集: 拟合 + 变换
    X_eval_poly  = poly.transform(X_eval)                         # 评估集: 仅变换

    # 线性回归拟合多项式特征
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_eval_poly)
    residuals = y_pred - y_eval

    mse = mean_squared_error(y_eval, y_pred)
    mse_list.append(mse)

    # ---- 子图 1: 所有阶数的拟合曲线叠加 ----
    plt.subplot(2, 2, 1)
    # 生成平滑曲线用于绘图 (200 个点)
    x_range = np.linspace(X_eval.min(), X_eval.max(), 200).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    y_range_pred = model.predict(x_range_poly)

    plt.scatter(X_eval, y_eval, alpha=0.4, s=10)                  # 实际数据点
    plt.plot(x_range, y_range_pred, label=f"Degree {degree}")

    plt.xlabel("X (GHz)")
    plt.ylabel("Y (GHz)")
    plt.title("Polynomial Regression Fit (Eval)")
    plt.legend()

    # ---- 子图 2: 各阶数的残差分布叠加 ----
    plt.subplot(2, 2, 2)
    plt.scatter(X_eval, residuals, alpha=0.5, s=10, label=f"Deg {degree}")
    plt.hlines(0, X_eval.min(), X_eval.max(), linestyles="--")
    plt.xlabel("X (GHz)")
    plt.ylabel("Residuals (GHz)")
    plt.title("Residual Plot")
    plt.legend()

# ---- 子图 3: MSE vs 阶数 (自动选择最优) ----
plt.subplot(2, 2, 3)
plt.plot(degrees, mse_list, marker="o")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("MSE vs Polynomial Degree")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "poly_eval_overview.png"))
plt.close()


# =============================================================================
# 图 2: 频率残差对比 (使用最优阶数)
# =============================================================================
best_degree = degrees[int(np.argmin(mse_list))]                   # MSE 最小的阶数
print(f"Best Polynomial Degree: {best_degree}")

# 用最优阶数重新训练
poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train)
X_eval_poly  = poly.transform(X_eval)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_eval_poly)

meas_residuals = X_eval - y_eval                                  # 校准前
pred_residuals = y_pred - y_eval                                  # 校准后

mse = mean_squared_error(y_pred, y_eval)
mae = np.mean(np.abs(pred_residuals)) * 1000                      # MHz
rmse = np.sqrt(mse) * 1000                                        # MHz

print(f"Eval MAE: {mae:.4f} MHz")
print(f"Eval MSE: {mse:.6f} GHz^2")
print(f"Eval RMSE: {rmse:.4f} MHz")

maxae = np.max(np.abs(pred_residuals)) * 1000
print(f"Eval MaxAE: {maxae:.4f} MHz")

# 按频率升序绘制残差对比图
x_freq = X_eval.squeeze()
idx = np.argsort(x_freq)

plt.figure(figsize=(8, 5))
plt.plot(
    x_freq[idx],
    meas_residuals.squeeze()[idx] * 1000,
    label="MEASURED ERROR"
)
plt.plot(
    x_freq[idx],
    pred_residuals.squeeze()[idx] * 1000,
    label=f"PREDICTED ERROR (deg={best_degree})"
)

plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residuals (MHz)")
plt.title(f"Polynomial Regression Residuals (Degree {best_degree})")
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
plt.savefig(
    os.path.join(
        Config.RESULT_SAVE_DIR,
        f"residual_plot_Polynomial_deg{best_degree}.png"
    )
)
plt.close()
