"""
=============================================================================
模块名称: KNN 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用 K 近邻回归 (KNeighborsRegressor) 作为基线模型。
  KNN 是一种非参数方法, 预测值为 K 个最近邻样本的目标值加权平均。

算法原理:
  对每个测试样本, 找到训练集中距离最近的 K 个样本,
  以它们的真实频率的加权平均 (uniform 或 distance-weighted) 作为预测值。
  优点: 无需训练过程、自然支持非线性、实现简单。
  缺点: 推理时需遍历全部训练集、高维时距离度量失效。

超参数搜索空间:
  - n_neighbors: 邻居数 [3, 5, 7, 9, 15, 25]
  - weights:     权重策略 [uniform (等权), distance (距离倒数加权)]
  - p:           距离度量 [1=曼哈顿, 2=欧几里得]

预处理: StandardScaler 标准化, 确保距离度量不受特征尺度影响

使用方式:
  cd compare/0-40ghz/knn && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置"""
    MODEL_TYPE = "KNN"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_csv(path):
    """
    读取两列 CSV 频率数据

    CSV 格式: 第1行表头, 第1列 X (测量频率 GHz), 第2列 y (真实频率 GHz)
    """
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1)
    )
    X = data[:, 0:1]                                              # 频率 / 特征
    y = data[:, 1:2]                                              # 残差 / 目标
    return X, y


X_train, y_train = load_csv("./input/20251216/train.csv")
X_eval,  y_eval  = load_csv("./input/20251216/eval.csv")


# =============================================================================
# 2. KNN Pipeline: StandardScaler + KNN + GridSearchCV
# =============================================================================
# 使用 Pipeline 确保标准化步骤嵌入交叉验证, 防止数据泄露
pipe = Pipeline([
    ("scaler", StandardScaler()),                                 # 标准化: 均值 0, 标准差 1
    ("knn", KNeighborsRegressor())                                # KNN 回归器
])

# 网格搜索参数空间
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 15, 25],                    # 邻居数量
    "knn__weights": ["uniform", "distance"],                      # uniform=等权, distance=距离倒数加权
    "knn__p": [1, 2],                                             # 1=曼哈顿距离, 2=欧几里得距离
}

# 5 折交叉验证搜索
search = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train.ravel())

print("Best KNN params:", search.best_params_)
print("Best CV MSE:", -search.best_score_)

model = search.best_estimator_


# =============================================================================
# 3. 预测与误差评估
# =============================================================================
y_pred = model.predict(X_eval).reshape(-1, 1)

# 原始测量误差 vs 模型修正后残差
meas_residuals = X_eval - y_eval
pred_residuals = y_pred - y_eval

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

# 测量误差 vs KNN 校正误差
plt.plot(
    x_freq[idx],
    meas_residuals.squeeze()[idx] * 1000,
    label="MEASURED ERROR"
)
plt.plot(
    x_freq[idx],
    pred_residuals.squeeze()[idx] * 1000,
    label="KNN CALIBRATED ERROR"
)

plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residuals (MHz)")
plt.title("KNN Frequency Residuals")
plt.grid(True)

meas_mae = np.mean(np.abs(meas_residuals)) * 1000
pred_mae = np.mean(np.abs(pred_residuals)) * 1000

plt.text(
    0.05, 0.95,
    f"MEAS_MAE: {meas_mae:.4f} MHz\nKNN_MAE: {pred_mae:.4f} MHz",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "knn_residuals.png"))
plt.close()
