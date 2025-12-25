import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ============================================================
# Config
# ============================================================
class Config:
    MODEL_TYPE = "PolynomialRegression"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. 读取 CSV 数据
# ============================================================
def load_csv(path):
    """
    CSV:
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
    X = data[:, 0:1]
    y = data[:, 1:2]
    return X, y


X_train, y_train = load_csv("./input/20251216/train.csv")
X_eval,  y_eval  = load_csv("./input/20251216/eval.csv")


# ============================================================
# 2. 多项式阶数设置
# ============================================================
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# ============================================================
# 图 1：不同阶数拟合 + 残差 + MSE
# ============================================================
plt.figure(figsize=(12, 10))

mse_list = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_eval_poly  = poly.transform(X_eval)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_eval_poly)
    residuals = y_pred - y_eval

    mse = mean_squared_error(y_eval, y_pred)
    mse_list.append(mse)

    # 子图1：拟合曲线
    plt.subplot(2, 2, 1)
    x_range = np.linspace(X_eval.min(), X_eval.max(), 200).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    y_range_pred = model.predict(x_range_poly)

    plt.scatter(X_eval, y_eval, alpha=0.4, s=10)
    plt.plot(x_range, y_range_pred, label=f"Degree {degree}")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polynomial Regression Fit (Eval)")
    plt.legend()

    # 子图2：残差图
    plt.subplot(2, 2, 2)
    plt.scatter(X_eval, residuals, alpha=0.5, s=10, label=f"Deg {degree}")
    plt.hlines(0, X_eval.min(), X_eval.max(), linestyles="--")
    plt.xlabel("X")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.legend()

# 子图3：MSE vs 阶数
plt.subplot(2, 2, 3)
plt.plot(degrees, mse_list, marker="o")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("MSE vs Polynomial Degree")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, "poly_eval_overview.png"))
plt.close()


# ============================================================
# 图 2：频率残差对比（选最佳阶数）
# ============================================================
best_degree = degrees[int(np.argmin(mse_list))]
print(f"Best Polynomial Degree: {best_degree}")

poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train)
X_eval_poly  = poly.transform(X_eval)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_eval_poly)

meas_residuals = X_eval - y_eval
pred_residuals = y_pred - y_eval

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
