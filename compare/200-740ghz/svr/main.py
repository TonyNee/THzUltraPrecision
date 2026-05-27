"""
=============================================================================
模块名称: SVR 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用支持向量回归 (SVR) 作为基线模型。
  SVR 通过寻找一个 ε-不敏感管道, 使得大部分训练样本落在管道内,
  同时最小化管道外的偏差, 具有很强的非线性拟合能力。

算法原理:
  使用 RBF (高斯径向基) 核函数将数据映射到高维空间:
    K(x, x') = exp(-γ||x - x'||^2)
  核心超参数:
    C:     正则化参数, 控制对管道外样本的惩罚力度
    γ:     RBF 核宽度, 控制单个样本的影响范围
    ε:     管道宽度, 管道内的样本不计入损失

预处理: StandardScaler 标准化 → SVR (RBF kernel) → RandomizedSearchCV

使用方式:
  cd compare/0-40ghz/svr && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from scipy.stats import loguniform


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置"""
    MODEL_TYPE = "SVR_RBF_SCALED"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_csv(path):
    """
    读取两列 CSV 频率数据

    CSV 格式:
      第1行: 表头
      第1列: X (Frequency, GHz)
      第2列: Y (True Frequency, GHz)
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
# 2. Pipeline: StandardScaler + SVR (RBF) + RandomizedSearchCV
# =============================================================================
# Pipeline 确保标准化在每折 CV 都正确执行, 防止数据泄露
pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])

# 参数分布: 使用对数均匀分布 (loguniform), 因为 C/γ/ε 通常跨多个数量级
param_dist = {
    "svr__C": loguniform(1e0, 1e4),                               # 正则化参数: 1 ~ 10000
    "svr__gamma": loguniform(1e-4, 1e1),                           # RBF 核宽度: 0.0001 ~ 10
    "svr__epsilon": loguniform(1e-4, 1e-1),                        # ε-管道宽度: 0.0001 ~ 0.1
}

# 随机搜索: 40 组参数, 5 折 CV
search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=40,
    scoring="neg_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
search.fit(X_train, y_train.ravel())

# 备选: 不使用 Pipeline 的写法 (保留供参考)
# svr = SVR(kernel="rbf")
# param_dist = {
#     "C": loguniform(1e0, 1e4),
#     "gamma": loguniform(1e-4, 1e0),
#     "epsilon": loguniform(1e-4, 1e-1)
# }
# search = RandomizedSearchCV(
#     estimator=svr,
#     param_distributions=param_dist,
#     n_iter=40,
#     scoring="neg_mean_squared_error",
#     cv=5,
#     random_state=42,
#     n_jobs=-1,
#     verbose=2
# )
# search.fit(X_train, y_train.ravel())

# =============================================================================
# 3. 最优模型 & 评估
# =============================================================================
best_model = search.best_estimator_

print("Best SVR params:", search.best_params_)
print("Best CV MSE:", -search.best_score_)

y_pred = best_model.predict(X_eval).reshape(-1, 1)

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


# =============================================================================
# 图: 频率残差对比图 (MHz)
# =============================================================================
x_freq = X_eval.squeeze()
idx = np.argsort(x_freq)                                          # 按频率升序

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
plt.title("SVR (RBF + StandardScaler) Frequency Residuals")
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
        "residual_plot_SVR_RBF_SCALED.png"
    )
)
plt.close()
