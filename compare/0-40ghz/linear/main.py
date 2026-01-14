import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# -------------------------------
# Config 类
# -------------------------------
class Config:
    MODEL_TYPE = "LinearRegression"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------------
# 1. 读取 CSV 数据
# -------------------------------
def load_csv(path):
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
X_eval, y_eval   = load_csv("./input/20251216/eval.csv")


# -------------------------------
# 2. 训练线性回归模型
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)


# -------------------------------
# 3. 预测 & 残差
# -------------------------------
y_pred = model.predict(X_eval)

meas_residuals = X_eval - y_eval      # 校准前残差
pred_residuals = y_pred - y_eval      # 校准后残差

mse = mean_squared_error(y_pred, y_eval)
print(f"Eval MSE: {mse:.6f}")


# ============================================================
# 图 1：2×2 分析图
# ============================================================
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.scatter(X_eval, y_eval, alpha=0.6, label="Actual Data")
plt.plot(X_eval, y_pred, linewidth=2, label="Regression Line")
plt.title("Scatter Plot with Regression Line (Eval)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(X_eval, pred_residuals, alpha=0.6)
plt.hlines(0, X_eval.min(), X_eval.max(), linestyles="--")
plt.title("Residual Plot (Eval)")
plt.xlabel("X")
plt.ylabel("Residuals")

plt.subplot(2, 2, 3)
plt.scatter(y_eval, y_pred, alpha=0.6)
min_y, max_y = y_eval.min(), y_eval.max()
plt.plot([min_y, max_y], [min_y, max_y], linestyle="--", label="Perfect Fit")
plt.title("Predicted vs Actual (Eval)")
plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.legend()

plt.subplot(2, 2, 4)
plt.hist(pred_residuals, bins=20, edgecolor="black", alpha=0.7)
plt.title("Residuals Distribution (Eval)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "linear_eval_overview.png"))
plt.close()


# ============================================================
# 图 2：频率残差对比图
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
