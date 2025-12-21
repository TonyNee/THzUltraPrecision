import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from model_resmlp import HighPrecCalibrator

# ============================
# 配置
# ============================
EVAL_CSV = '../data/20250928/THz_eval_20250928.csv'
MODEL_PATH = './model/resmlp_calibration_best.pth'
SAVE_DIR = "./result_resmlp/"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================
# 1. 加载数据
# ============================
df = pd.read_csv(EVAL_CSV)
x_eval = df.iloc[:, 0].values.astype(np.float32)
y_eval = df.iloc[:, 1].values.astype(np.float32)

X = torch.tensor(x_eval.reshape(-1, 1)).to(device)
Y = torch.tensor(y_eval.reshape(-1, 1)).to(device)

# ============================
# 2. 加载模型
# ============================
model = HighPrecCalibrator().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ============================
# 3. 预测
# ============================
with torch.no_grad():
    y_pred = model(X).cpu().numpy()

y_true = Y.cpu().numpy()

# ============================
# 4. 误差指标
# ============================
residuals = (y_true - y_pred).flatten()

MAE = np.mean(np.abs(residuals)) * 1000      # MHz
STD = np.std(residuals) * 1000               # MHz
MSE = np.mean(residuals ** 2)                # GHz^2

print("\n===== ResMLP 评估结果 =====")
print(f"MAE: {MAE:.3f} MHz")
print(f"STD: {STD:.3f} MHz")
print(f"MSE: {MSE:.10f} GHz^2")

# ============================
# 5. 保存 CSV
# ============================
df_out = pd.DataFrame({
    "F_test_GHz": x_eval,
    "F_true_GHz": y_eval,
    "F_pred_GHz": y_pred.flatten(),
    "Residual(GHz)": residuals
})
df_out.to_csv(os.path.join(SAVE_DIR, "resmlp_predictions.csv"), index=False)

# ============================
# 6. 残差图
# ============================
plt.figure(figsize=(8,5))
plt.plot(residuals * 1000)
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Sample Index")
plt.ylabel("Residual (MHz)")
plt.title("ResMLP Frequency Calibration Residuals")
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "residual_plot.png"))
plt.close()
