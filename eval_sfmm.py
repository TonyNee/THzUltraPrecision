import os
import pandas as pd
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import time

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
start_time = time.perf_counter()
with torch.no_grad():
    y_pred = model(X).cpu().numpy()
end_time = time.perf_counter()
total_time_ms = (end_time - start_time) * 1000  # 转换为毫秒
avg_time_us = (end_time - start_time) / len(X) * 1e6  # 转换为微秒
print("\n===== 推理时间 =====")
print(f"总样本数: {len(X)}")
print(f"总推理时间: {total_time_ms:.3f} ms")
print(f"平均每样本: {avg_time_us:.3f} μs ({avg_time_us/1000:.5f} ms)")

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

metrics = {"MAE": MAE, "MSE": MSE, "RMSE": RMSE}
Config.update_yaml(model_dir=args.mdir, metrics_dict=metrics)

print("\n===== ResMLP 评估结果 =====")
print(f"MAE: {MAE:.10f} MHz")
print(f"MSE: {MSE:.10f} GHz^2")
print(f"RMSE: {RMSE:.10f} MHz\n")

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
# 6. 绘制测量值和预测值的直方图
# ============================
true_value = 200

# 创建子图布局
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'{Config.MODEL_TYPE} Model Evaluation Visualization', fontsize=16)

# ================= Figure 1: 测量值直方图 =================
axes[0].hist(x_eval, bins=20, facecolor='blue', edgecolor='black', alpha=0.7, label='Measured Values')
axes[0].axvline(x=true_value, color='red', linestyle='--', linewidth=2, label=f'True Value = {true_value} GHz')
axes[0].axvline(x=np.mean(x_eval), color='green', linestyle='-', linewidth=2, 
               label=f'Meas Mean = {np.mean(x_eval):.6f} GHz')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Frequency (GHz)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Measured Values Distribution Histogram', fontsize=14)
axes[0].legend(loc='best', fontsize=10)

# ================= Figure 2: 预测值直方图 =================
axes[1].hist(y_pred.flatten(), bins=20, facecolor='orange', edgecolor='black', alpha=0.7, label='Predicted Values')
axes[1].axvline(x=true_value, color='red', linestyle='--', linewidth=2, label=f'True Value = {true_value} GHz')
axes[1].axvline(x=np.mean(y_pred.flatten()), color='green', linestyle='-', linewidth=2, 
               label=f'Pred Mean = {np.mean(y_pred.flatten()):.6f} GHz')
axes[1].set_xlabel('Frequency (GHz)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Predicted Values Distribution Histogram', fontsize=14)
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

# 调整布局并保存图表
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, f'evaluation_visualization_{model_name}.png'), dpi=300, bbox_inches='tight')
plt.show()