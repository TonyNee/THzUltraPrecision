import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from scipy.stats import loguniform


# ============================================================
# Config
# ============================================================
class Config:
    MODEL_TYPE = "SVR_RBF_SCALED"
    RESULT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. 读取 CSV
# ============================================================
def load_csv(path):
    """
    CSV:
        第1列: X (Frequency)
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
# 2. Pipeline：StandardScaler + SVR
# ============================================================
pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
param_dist = {
    "svr__C": loguniform(1e0, 1e4),
    "svr__gamma": loguniform(1e-4, 1e1),
    "svr__epsilon": loguniform(1e-4, 1e-1),
}
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


# svr = SVR(kernel="rbf")
# param_dist = {
#     "C": loguniform(1e0, 1e4),         # 1 ~ 10000
#     "gamma": loguniform(1e-4, 1e0),    # 0.0001 ~ 1
#     "epsilon": loguniform(1e-4, 1e-1)  # 0.0001 ~ 0.1
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

# ============================================================
# 4. 最优模型 & 评估
# ============================================================
best_model = search.best_estimator_

print("Best SVR params:", search.best_params_)
print("Best CV MSE:", -search.best_score_)

y_pred = best_model.predict(X_eval).reshape(-1, 1)

meas_residuals = X_eval - y_eval
pred_residuals = y_pred - y_eval

eval_mse = mean_squared_error(y_eval, y_pred)
print(f"Eval MSE: {eval_mse:.6f}")


# ============================================================
# 5. 频率残差对比图（MHz）
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
