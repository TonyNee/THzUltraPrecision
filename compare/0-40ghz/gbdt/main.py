"""
=============================================================================
模块名称: Gradient Boosting 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用梯度提升回归 (GradientBoostingRegressor) 作为基线模型。
  梯度提升是一种集成学习方法, 通过逐步添加弱学习器 (决策树),
  每一步拟合前一步的残差, 从而逐步减少整体误差。

算法原理:
  1. 从一个常数预测值开始 (通常是均值)
  2. 计算当前模型的残差
  3. 训练一棵浅层决策树来拟合残差
  4. 将新树以一定学习率添加到模型中
  5. 重复步骤 2-4, 共 n_estimators 轮

超参数搜索 (RandomizedSearchCV):
  - n_estimators:     100~400 (树的数量)
  - learning_rate:    0.01~0.16 (每棵树的贡献权重)
  - max_depth:        2~4 (单棵树深度, 浅树防过拟合)
  - subsample:        0.6~1.0 (每棵树使用的样本比例)
  - min_samples_leaf: 1~19 (叶节点最小样本数)

使用方式:
  cd compare/0-40ghz/gbdt && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置"""
    MODEL_TYPE = "GradientBoosting"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_csv(path):
    """
    读取两列 CSV 频率数据

    CSV 格式:
      第1行: 表头
      第1列: 频率测量值 (Measured Frequency, GHz)
      第2列: 频率真实值 (True Frequency, GHz)
    """
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1)
    )
    X = data[:, 0:1]                                              # 测量值
    y = data[:, 1:2]                                              # 真实值
    return X, y


X_train, y_train = load_csv("./input/20251216/train.csv")
X_eval,  y_eval  = load_csv("./input/20251216/eval.csv")


# =============================================================================
# 2. 梯度提升回归 + 随机搜索
# =============================================================================
gbr = GradientBoostingRegressor(random_state=42)

# 随机搜索参数分布: 连续参数用均匀分布采样
param_dist = {
    "n_estimators": randint(100, 400),                            # 树的数量 (迭代轮数)
    "learning_rate": uniform(0.01, 0.15),                         # 学习率 0.01 ~ 0.16
    "max_depth": randint(2, 5),                                   # 单棵树深度 2 ~ 4
    "subsample": uniform(0.6, 0.4),                               # 子采样比例 0.6 ~ 1.0
    "min_samples_leaf": randint(1, 20),                           # 叶节点最小样本数
}

# 随机搜索: 从参数分布中采样 n_iter=40 组, 5 折 CV
search = RandomizedSearchCV(
    estimator=gbr,
    param_distributions=param_dist,
    n_iter=40,                                                    # 随机采样 40 组参数组合
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train.ravel())

print("Best GBDT params:", search.best_params_)
print("Best CV MSE:", -search.best_score_)

model = search.best_estimator_


# =============================================================================
# 3. 预测与误差
# =============================================================================
y_pred = model.predict(X_eval).reshape(-1, 1)

# 残差定义
meas_residuals = X_eval - y_eval                                  # 校准前: 测量误差
pred_residuals = y_pred - y_eval                                  # 校准后: 模型修正残差

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
plt.title("Gradient Boosting Frequency Residuals")
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
        "residual_plot_GradientBoosting.png"
    )
)
plt.close()
