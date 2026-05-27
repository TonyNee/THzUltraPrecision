"""
=============================================================================
模块名称: Random Forest 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用随机森林回归 (RandomForestRegressor) 作为基线模型。
  随机森林通过构建多棵决策树并取平均来降低方差, 是一种 Bagging 集成方法。

算法原理:
  1. 从训练集中 Bootstrap 采样 N 个子集
  2. 对每个子集训练一棵决策树, 每次分裂仅考虑随机子集的特征
  3. 预测时取所有树的预测值平均

  双重随机性 (样本 + 特征) 降低了树间相关性, 有效减少过拟合。

超参数搜索 (RandomizedSearchCV):
  - n_estimators:     100~400 (树的数量, 越多越稳定)
  - max_depth:        [None, 5, 10, ..., 30] (单棵树最大深度)
  - min_samples_split: 2~9 (内部节点分裂最小样本数)
  - min_samples_leaf:  1~7 (叶节点最小样本数)
  - max_features:      ["sqrt", "log2", 1.0] (每棵树可用的特征比例)

使用方式:
  cd compare/0-40ghz/rf && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置"""
    MODEL_TYPE = "RandomForest"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_csv(path):
    """
    读取两列 CSV 频率数据: 第1列 X (测量频率 GHz), 第2列 y (真实频率 GHz)
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
# 2. 随机森林 + RandomizedSearchCV
# =============================================================================
# 基础随机森林模型: n_jobs=-1 并行训练所有树
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# 参数分布
param_dist = {
    "n_estimators": randint(100, 400),                            # 树的数量
    "max_depth": [None, 5, 10, 15, 20, 25, 30],                   # 树深度 (None=不限制)
    "min_samples_split": randint(2, 10),                          # 内部节点最少样本数
    "min_samples_leaf": randint(1, 8),                            # 叶节点最少样本数
    "max_features": ["sqrt", "log2", 1.0],                        # 分裂时考虑的特征比例
}

# 随机搜索: n_iter=30 组, 5 折 CV
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train.ravel())

print("\n================ Best RF Params ================")
for k, v in search.best_params_.items():
    print(f"{k}: {v}")
print(f"Best CV MSE: {-search.best_score_:.6f}")

model = search.best_estimator_


# =============================================================================
# 3. 预测与残差
# =============================================================================
y_pred = model.predict(X_eval).reshape(-1, 1)

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
    label="PREDICTED ERROR"
)

plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residuals (MHz)")
plt.title("RandomForest Frequency Residuals")
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
        "residual_plot_RandomForest.png"
    )
)
plt.close()
