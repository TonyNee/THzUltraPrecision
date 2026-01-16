import os
import pandas as pd
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

from config import Config, Utils


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
x_freq = y_eval
idx = np.argsort(x_freq)

save_name = f"residual_plot_{model_name}.png"
title_str = f"{Config.MODEL_TYPE} Frequency Residuals"

plt.figure(figsize=(10,6))
plt.plot(x_freq[idx], meas_residuals[idx] * 1000, label='MEASURED ERROR')
plt.plot(x_freq[idx], pred_residuals[idx] * 1000, label='PREDICTED ERROR')
plt.axhline(0, linestyle="--", color="black")
plt.axvline(747, linestyle=':', color='gray', alpha=0.7)
y_min, y_max = plt.ylim()
plt.text(747, y_min - (y_max - y_min) * 0.018, '747', ha='center', va='top', fontsize=10, color='gray', alpha=0.8)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residuals (MHz)")
plt.title(title_str)
plt.grid(True)

# 计算MAE
meas_mae = np.mean(np.abs(meas_residuals)) * 1000
pred_mae = np.mean(np.abs(pred_residuals)) * 1000

# 应用标注函数
meas_results = Utils.annotate_max_abs_by_range(x_freq[idx], meas_residuals[idx] * 1000, 'MEAS', 0)
pred_results = Utils.annotate_max_abs_by_range(x_freq[idx], pred_residuals[idx] * 1000, 'PRED', 1)

# 构建信息文本
# info_text = f'MEAS_MAE: {meas_mae:.2f} MHz, PRED_MAE: {pred_mae:.2f} MHz\n'
# meas_max_all = np.max(np.abs(meas_residuals[idx] * 1000))
# pred_max_all = np.max(np.abs(pred_residuals[idx] * 1000))
# info_text += f'MEAS_MAX: {meas_max_all:.2f} MHz, PRED_MAX: {pred_max_all:.2f} MHz'
info_text = "   Metric              MEAS         PRED\n"
info_text += "-" * 50 + "\n"
meas_max_all = np.max(np.abs(meas_residuals[idx] * 1000))
pred_max_all = np.max(np.abs(pred_residuals[idx] * 1000))
info_text += f"MAE (MHz)        {meas_mae:>7.2f}    {pred_mae:>7.2f}\n"
info_text += f"MAX (MHz)        {meas_max_all:>7.2f}    {pred_max_all:>7.2f}"
plt.text(0.015, 0.30, info_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(Config.RESULT_SAVE_DIR, save_name))
plt.close()


# ============================
# 7. 分通道MAE计算
# ============================
threshold = 747 

low_freq_mask = x_freq < threshold
high_freq_mask = x_freq > threshold
if np.any(low_freq_mask):
    low_freq_meas_residuals = meas_residuals[low_freq_mask]
    low_freq_pred_residuals = pred_residuals[low_freq_mask]
    
    low_freq_meas_mae = np.mean(np.abs(low_freq_meas_residuals)) * 1000  # MHz
    low_freq_pred_mae = np.mean(np.abs(low_freq_pred_residuals)) * 1000  # MHz
    
    low_freq_count = np.sum(low_freq_mask)  # 统计低频数据点数量
else:
    low_freq_meas_mae = 0
    low_freq_pred_mae = 0
    low_freq_count = 0

if np.any(high_freq_mask):
    high_freq_meas_residuals = meas_residuals[high_freq_mask]
    high_freq_pred_residuals = pred_residuals[high_freq_mask]
    
    high_freq_meas_mae = np.mean(np.abs(high_freq_meas_residuals)) * 1000  # MHz
    high_freq_pred_mae = np.mean(np.abs(high_freq_pred_residuals)) * 1000  # MHz
    
    high_freq_count = np.sum(high_freq_mask)  # 统计高频数据点数量
else:
    high_freq_meas_mae = 0
    high_freq_pred_mae = 0
    high_freq_count = 0

# 打印分通道结果
print("\n===== 分通道评估结果 =====")
print(f"CH1: 低频段 (Freq < {threshold} GHz, {low_freq_count} points):")
print(f"  MEAS_MAE: {low_freq_meas_mae:.4f} MHz")
print(f"  PRED_MAE: {low_freq_pred_mae:.4f} MHz")

print(f"\nCH2: 高频段 (Freq > {threshold} GHz, {high_freq_count} points):")
print(f"  MEAS_MAE: {high_freq_meas_mae:.4f} MHz")
print(f"  PRED_MAE: {high_freq_pred_mae:.4f} MHz")





'''
# 创建结果表格图片
table_data = [
    ['Frequency Band', 'Data Points', 'MEAS_MAE (MHz)', 'PRED_MAE (MHz)'],
    ['Low Freq (≤ 747 GHz)', f'{low_freq_count}', f'{low_freq_meas_mae:.4f}', f'{low_freq_pred_mae:.4f}'],
    ['High Freq (> 747 GHz)', f'{high_freq_count}', f'{high_freq_meas_mae:.4f}', f'{high_freq_pred_mae:.4f}'],
    ['Total', f'{len(x_freq)}', f'{meas_mae:.4f}', f'{pred_mae:.4f}']
]
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=table_data,
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.2, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2) 
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')
for i in range(1, len(table_data)):
    for j in range(len(table_data[i])):
        if i % 2 == 0: 
            table[(i, j)].set_facecolor('#f2f2f2')
        else: 
            table[(i, j)].set_facecolor('#ffffff')
for i in range(len(table_data)):
    for j in range(len(table_data[i])):
        table[(i, j)].set_edgecolor('black')
plt.title('Frequency Band Evaluation Results Summary', fontsize=14, fontweight='bold', pad=20)

table_save_name = os.path.join(Config.RESULT_SAVE_DIR, f'table_results_{model_name}.png')
plt.savefig(table_save_name, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n结果表格已保存至: {table_save_name}")

'''

