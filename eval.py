import os
import pandas as pd
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

from config import Config


# ============================
# 0. 配置环境
# ============================
parser = argparse.ArgumentParser()
parser.add_argument("--mdir", required=True)
args = parser.parse_args()
Config.load_yaml(args.mdir)
device = torch.device(Config.DEVICE)
print(f"Device: {device}")

# ============================
# 1. 加载数据
# ============================
df = pd.read_csv(Config.EVAL_CSV)
x_eval = df.iloc[:, 0].values.astype(np.float32)
y_eval = df.iloc[:, 1].values.astype(np.float32)

X = torch.tensor(x_eval.reshape(-1, 1)).to(device)
Y = torch.tensor(y_eval.reshape(-1, 1)).to(device)

# ============================
# 2. 加载模型
# ============================
model = Config.MODEL_CLASS().to(device)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
model.eval()

# ============================
# 3. 预测
# ============================
with torch.no_grad():
    y_pred = model(X).cpu().numpy()

y_true = Y.cpu().numpy()
x_true = X.cpu().numpy()

# ============================
# 4. 误差指标
# ============================
meas_residuals = (x_true - y_true).flatten()
pred_residuals = (y_pred - y_true).flatten()

MAE = float(np.mean(np.abs(pred_residuals)) * 1000)      # MHz
MSE = float(np.mean(pred_residuals ** 2))                # GHz^2
RMSE = float(np.sqrt(MSE) * 1000)                   # MHz
R2 = float(1 - np.sum(pred_residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

metrics = {"MAE": MAE, "MSE": MSE, "RMSE": RMSE, "R2": R2}
Config.update_yaml(model_dir=args.mdir, metrics_dict=metrics)

print("\n===== ResMLP 评估结果 =====")
print(f"MAE: {MAE:.10f} MHz")
print(f"MSE: {MSE:.10f} GHz^2")
print(f"RMSE: {RMSE:.10f} MHz")
print(f"R2: {R2:.10f}")

# ============================
# 5. 保存 CSV
# ============================
model_name = Config.MODEL_TYPE.replace(" ", "_")

df_out = pd.DataFrame({
    "F_test_GHz": x_eval,
    "F_true_GHz": y_eval,
    "F_pred_GHz": y_pred.flatten(),
    "Residual(MHz)": pred_residuals * 1000
})
df_out.to_csv(os.path.join(Config.RESULT_SAVE_DIR, f"data_predicted_{model_name}.csv"), index=False)

# ============================
# 6. 残差图
# ============================
x_freq = x_eval
idx = np.argsort(x_freq)

save_name = f"residual_plot_{model_name}.png"
title_str = f"{Config.MODEL_TYPE} Frequency Residuals"

plt.figure(figsize=(8,5))
plt.plot(x_freq[idx], meas_residuals[idx] * 1000, label='MEASURED ERROR')
plt.plot(x_freq[idx], pred_residuals[idx] * 1000, label='PREDICTED ERROR')
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residuals (MHz)")
plt.title(title_str)
plt.grid(True)
meas_mae = np.mean(np.abs(meas_residuals)) * 1000
pred_mae = np.mean(np.abs(pred_residuals)) * 1000
plt.text(0.05, 0.95, f'MEAS_MAE: {meas_mae:.4f} MHz\nPRED_MAE: {pred_mae:.4f} MHz',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, save_name))
plt.close()


