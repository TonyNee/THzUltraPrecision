import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ============================================================
# Config
# ============================================================
class Config:
    MODEL_TYPE = "LassoRegression"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. 读取 CSV 数据
# ============================================================
def load_csv(path):
    """
    CSV 格式:
        第1列: X
        第2列: Y
        第1行为表头
    """
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1)
    )
    X = data[:, 0:1]   # (N, 1)
    y = data[:, 1:2]   # (N, 1)
    return X, y


X_train, y_train = load_csv("./input/20251216/train.csv")
X_eval,  y_eval  = load_csv("./input/20251216/eval.csv")


# ============================================================
# 2. 标准化 + Lasso 训练
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled  = scaler.transform(X_eval)

alpha = 1e-3  # 可根据需要调节
lasso = Lasso(alpha=alpha, max_iter=10000)
lasso.fit(X_train_scaled, y_train.ravel())


# ============================================================
# 3. 预测 & 残差
# ============================================================
y_pred = lasso.predict(X_eval_scaled).reshape(-1, 1)

meas_residuals = X_eval - y_eval      # 校准前
pred_residuals = y_pred - y_eval      # 校准后

mse = mean_squared_error(y_eval, y_pred)
print(f"Eval MSE: {mse:.6f}")


# ============================================================
# 图 1：2×2 综合分析图
# ============================================================
plt.figure(figsize=(12, 10))

# 1️⃣ 散点 + 回归线
plt.subplot(2, 2, 1)
plt.scatter(X_eval, y_eval, alpha=0.6, label="Actual")
plt.plot(X_eval, y_pred, linewidth=2, label="Lasso Fit")
plt.title("Scatter Plot with Lasso Regression (Eval)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# 2️⃣ 残差图
plt.subplot(2, 2, 2)
plt.scatter(X_eval, pred_residuals, alpha=0.6)
plt.hlines(0, X_eval.min(), X_eval.max(), linestyles="--")
plt.title("Residual Plot (Eval)")
plt.xlabel("X")
plt.ylabel("Residuals")

# 3️⃣ Pred vs Actual
plt.subplot(2, 2, 3)
plt.scatter(y_eval, y_pred, alpha=0.6)
min_y, max_y = y_eval.min(), y_eval.max()
plt.plot([min_y, max_y], [min_y, max_y], linestyle="--", label="Perfect Fit")
plt.title("Predicted vs Actual")
plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.legend()

# 4️⃣ 残差直方图
plt.subplot(2, 2, 4)
plt.hist(pred_residuals, bins=20, edgecolor="black", alpha=0.7)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "lasso_eval_overview.png"))
plt.close()


# ============================================================
# 图 2：Lasso 系数条形图（稀疏性）
# ============================================================
plt.figure(figsize=(6, 4))
plt.bar([0], lasso.coef_, color="cyan")
plt.axhline(0, linestyle="--", color="black")
plt.title("Lasso Coefficient (L1 Sparsity)")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "lasso_coefficients.png"))
plt.close()


# ============================================================
# 图 3：频率残差对比图（MHz）
# ============================================================
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
