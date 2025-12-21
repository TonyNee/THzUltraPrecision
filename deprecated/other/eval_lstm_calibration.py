import numpy as np
import pandas as pd
import torch
from lstm_model import LSTMCalibNet

SEQ_LEN = 16
EVAL_CSV = "./data/THz_eval_20250928.csv"

# -----------------------
# 加载 eval 数据
# -----------------------
df = pd.read_csv(EVAL_CSV)
x = df["Fexperiment_GHz"].values.astype(np.float32)
y = df["Fstandard_GHz"].values.astype(np.float32)

# -----------------------
# 加载 norm 参数
# -----------------------
norm = np.load("./model/lstm_norm_params.npz")
x_mean, x_std = norm["x_mean"], norm["x_std"]
r_mean, r_std = norm["r_mean"], norm["r_std"]

# 构造序列
def build_seq(arr, seq_len):
    Xs = []
    for i in range(len(arr) - seq_len):
        Xs.append(arr[i:i+seq_len])
    return np.array(Xs)

res = y - x
X_seq = build_seq(x, SEQ_LEN)
R_true = build_seq(res, SEQ_LEN)

# 归一化
Xn = (X_seq - x_mean) / (x_std + 1e-12)

# -----------------------
# 加载模型
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMCalibNet()
model.load_state_dict(torch.load("./model/lstm_calibration_best.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------
# 推理
# -----------------------
with torch.no_grad():
    pred_norm = model(torch.tensor(Xn).float().unsqueeze(-1).to(device)).cpu().numpy()

pred_res = pred_norm * r_std + r_mean

# 评估
residuals = R_true[:, -1] - pred_res[:, -1]

mae = np.mean(np.abs(residuals)) * 1000
std = np.std(residuals) * 1000
mse = np.mean((residuals)**2)
print("\n===== LSTM 评估结果 =====")
print(f"MAE: {mae:.6f} MHz")
print(f"STD: {std:.6f} MHz")
print(f"MSE: {mse:.10f} GHz^2")
