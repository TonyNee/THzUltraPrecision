import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


# ============================================================
# Config
# ============================================================
class Config:
    MODEL_TYPE = "GradientBoosting"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. 读取 CSV
# ============================================================
def load_csv(path):
    """
    CSV:
        第1列: 频率测量值 (Measured Frequency, GHz)
        第2列: 频率真实值 (True Frequency, GHz)
        第1行为表头
    """
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1)
    )
    X = data[:, 0:1]   # measured
    y = data[:, 1:2]   # true
    return X, y


X_train, y_train = load_csv("./input/20251216/train.csv")
X_eval,  y_eval  = load_csv("./input/20251216/eval.csv")


# ============================================================
# 2. Gradient Boosting 回归
# ============================================================
gbr = GradientBoostingRegressor(random_state=42)

param_dist = {
    "n_estimators": randint(100, 400),
    "learning_rate": uniform(0.01, 0.15),     # 0.01 ~ 0.16
    "max_depth": randint(2, 5),                # 2 ~ 4
    "subsample": uniform(0.6, 0.4),            # 0.6 ~ 1.0
    "min_samples_leaf": randint(1, 20),
}

search = RandomizedSearchCV(
    estimator=gbr,
    param_distributions=param_dist,
    n_iter=40,
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


# ============================================================
# 3. 预测 & 误差
# ============================================================
y_pred = model.predict(X_eval).reshape(-1, 1)

# 残差定义（与你前面完全一致）
meas_residuals = X_eval - y_eval        # 测量误差
pred_residuals = y_pred - y_eval        # 预测后误差

eval_mse = mean_squared_error(y_eval, y_pred)
print(f"Eval MSE: {eval_mse:.6f}")


# ============================================================
# 4. 频率残差对比图（MHz）
# ============================================================
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
