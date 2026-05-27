"""
=============================================================================
模块名称: Decision Tree 基线模型
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (compare 基线对比)
作　　者: TonyNee
创建日期: 2025-12-16
最后修改: 2026-05-27
=============================================================================

功能概述:
  使用决策树回归 (DecisionTreeRegressor) 作为基线模型, 与神经网络方法对比。
  通过网格搜索 (GridSearchCV) 优化决策树的关键超参数, 评估在校准任务上的表现。

算法原理:
  决策树通过递归划分特征空间来拟合数据。在频率校准任务中,
  它以测量频率为输入, 学习从测量值到真实值的映射关系。
  优点: 可解释性强、无需特征缩放、能捕捉非线性关系。
  缺点: 容易过拟合、对数据微小变化敏感。

超参数搜索空间:
  - max_depth:        树的最大深度 [3, 5, 7, 9, 12, None]
  - min_samples_leaf: 叶节点最小样本数 [1, 3, 5, 10]
  - min_samples_split: 内部节点分裂最小样本数 [2, 5, 10]
  - ccp_alpha:        最小代价复杂度剪枝参数 [0.0, 1e-5, 1e-4, 1e-3]

输出内容:
  - 最佳超参数和 CV MSE
  - 评估集 MAE / MSE / RMSE / MaxAE
  - 三子图综合分析图: 特征-目标散点图、实际值-预测值对比、残差图
  - 频率残差对比图 (测量误差 vs 模型修正误差, MHz 单位)

使用方式:
  cd compare/0-40ghz/dt && python main.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# =============================================================================
# Config 类 — 实验配置
# =============================================================================
class Config:
    """本实验的配置 (模型类型 + 输出目录)"""
    MODEL_TYPE = "DecisionTree"                                   # 模型标识
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))  # 结果保存到脚本所在目录


# =============================================================================
# 1. 数据加载
# =============================================================================
def load_csv(path):
    """
    读取两列 CSV 频率数据

    CSV 格式:
      第1行: 表头 (跳过)
      第1列: 频率测量值 X (Measured Frequency, GHz)
      第2列: 频率真实值 Y (True Frequency, GHz)

    参数:
      path: CSV 文件路径

    返回:
      X: 形状 (N, 1) 的测量频率数组
      y: 形状 (N, 1) 的真实频率数组
    """
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,                                               # 跳过表头行
        usecols=(0, 1)                                            # 只读取前两列
    )
    X = data[:, 0:1]                                              # (N, 1) 测量频率
    y = data[:, 1:2]                                              # (N, 1) 真实频率
    return X, y


# 加载训练集与评估集
X_train, y_train = load_csv("./input/20251216/train.csv")
X_eval,  y_eval  = load_csv("./input/20251216/eval.csv")


# =============================================================================
# 2. 决策树 + 网格搜索 (GridSearchCV)
# =============================================================================
# 基础模型: 固定随机种子保证可复现
base_model = DecisionTreeRegressor(
    random_state=42
)

# 网格搜索参数空间
param_grid = {
    "max_depth": [3, 5, 7, 9, 12, None],                         # 树深度 (None=不限)
    "min_samples_leaf": [1, 3, 5, 10],                            # 叶节点最小样本数 (防过拟合)
    "min_samples_split": [2, 5, 10],                              # 分裂最小样本数
    "ccp_alpha": [0.0, 1e-5, 1e-4, 1e-3],                        # 代价复杂度剪枝 (α 越大越简单)
}

# 5 折交叉验证网格搜索, 以负 MSE 为评分标准
search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,                                                         # 5 折交叉验证
    scoring="neg_mean_squared_error",                             # 负均方误差 (越大越好)
    n_jobs=-1,                                                    # 并行使用所有 CPU 核心
    verbose=1
)

# 在训练集上执行搜索 (y 需要展平为 1D)
search.fit(X_train, y_train.ravel())

print("Best DT params:", search.best_params_)
print("Best CV MSE:", -search.best_score_)                        # 取负号恢复正 MSE

# 提取最优模型
model = search.best_estimator_


# =============================================================================
# 3. 预测与误差评估
# =============================================================================
# 在评估集上进行预测
y_pred = model.predict(X_eval).reshape(-1, 1)

# 残差定义:
#   meas_residuals = 测量值 - 真实值  (原始测量偏差, 校准前)
#   pred_residuals = 预测值 - 真实值  (模型修正后残差, 校准后)
meas_residuals = X_eval - y_eval
pred_residuals = y_pred - y_eval

# 计算标准回归指标 (单位换算: GHz^2 -> MHz 需乘以 10^6)
mse = mean_squared_error(y_pred, y_eval)
mae = np.mean(np.abs(pred_residuals)) * 1000                      # 转换为 MHz
rmse = np.sqrt(mse) * 1000                                        # 转换为 MHz

print(f"Eval MAE: {mae:.4f} MHz")
print(f"Eval MSE: {mse:.6f} GHz^2")
print(f"Eval RMSE: {rmse:.4f} MHz")

maxae = np.max(np.abs(pred_residuals)) * 1000                     # 最大绝对误差 (MHz)
print(f"Eval MaxAE: {maxae:.4f} MHz")


# =============================================================================
# 图 1: 三子图综合分析 (特征-目标 / 实际-预测 / 残差)
# =============================================================================
plt.figure(figsize=(8, 12))

# ---- 子图 1: 特征 vs 目标散点图 ----
plt.subplot(3, 1, 1)
plt.scatter(X_train, y_train, alpha=0.6, label="Train")           # 训练集分布
plt.scatter(X_eval, y_eval, alpha=0.6, label="Eval")              # 评估集分布
plt.title("Feature vs Target")
plt.xlabel("Frequency (GHz)")
plt.ylabel("True Frequency (GHz)")
plt.legend()

# ---- 子图 2: 实际值 vs 预测值 ----
plt.subplot(3, 1, 2)
plt.scatter(y_eval, y_pred, alpha=0.6)
min_y, max_y = y_eval.min(), y_eval.max()
plt.plot([min_y, max_y], [min_y, max_y], linestyle="--", color="black")  # 完美预测对角线
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# ---- 子图 3: 残差分布图 ----
plt.subplot(3, 1, 3)
plt.scatter(X_eval, pred_residuals, alpha=0.6)
plt.axhline(0, linestyle="--", color="black")                     # 零残差参考线
plt.title("Residuals (Eval)")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residuals (GHz)")

plt.tight_layout()
plt.savefig(
    os.path.join(
        Config.RESULT_SAVE_DIR,
        "decision_tree_eval_overview.png"
    )
)
plt.close()


# =============================================================================
# 图 2: 频率残差对比图 (MHz 单位)
# =============================================================================
# 按频率升序排列以绘制连续的残差曲线
x_freq = X_eval.squeeze()
idx = np.argsort(x_freq)

plt.figure(figsize=(8, 5))

# 绘制测量误差 (蓝色) 和模型修正后误差 (橙色)
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

plt.axhline(0, linestyle="--", color="black")                     # 零误差参考线
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residuals (MHz)")
plt.title("Decision Tree Frequency Residuals")
plt.grid(True)

# 计算并显示测量/预测的 MAE
meas_mae = np.mean(np.abs(meas_residuals)) * 1000
pred_mae = np.mean(np.abs(pred_residuals)) * 1000

plt.text(
    0.05, 0.95,
    f"MEAS_MAE: {meas_mae:.4f} MHz\nPRED_MAE: {pred_mae:.4f} MHz",
    transform=plt.gca().transAxes,                                # 轴坐标系统
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(
        Config.RESULT_SAVE_DIR,
        "residual_plot_DecisionTree.png"
    )
)
plt.close()
