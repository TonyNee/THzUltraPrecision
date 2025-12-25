import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# ============================================================
# Config
# ============================================================
class Config:
    MODEL_TYPE = "RandomForest"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. 读取 CSV
# ============================================================
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
X_eval,  y_eval  = load_csv("./input/20251216/eval.csv")


# ============================================================
# 2. RandomForest + RandomizedSearchCV
# ============================================================
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

param_dist = {
    "n_estimators": randint(100, 400),
    "max_depth": [None, 5, 10, 15, 20, 25, 30],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 8),
    "max_features": ["sqrt", "log2", 1.0],
}

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


# ============================================================
# 3. 预测 & 残差
# ============================================================
y_pred = model.predict(X_eval).reshape(-1, 1)

meas_residuals = X_eval - y_eval
pred_residuals = y_pred - y_eval

mse = mean_squared_error(y_eval, y_pred)
print(f"Eval MSE: {mse:.6f}")


# ============================================================
# 图：频率残差对比（MHz）
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
