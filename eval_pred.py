import os
import pandas as pd
import numpy as np
import torch
import argparse

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
df = pd.read_csv(Config.EVAL_CSV, header=None)
x_eval = df.iloc[:, 0].values.astype(np.float32)
X = torch.tensor(x_eval.reshape(-1, 1)).to(device)

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


# ============================
# 4. 保存 CSV
# ============================
model_name = Config.MODEL_TYPE.replace(" ", "_")

df_out = pd.DataFrame({
    "F_test_GHz": x_eval,
    "F_pred_GHz": y_pred.flatten(),
})
df_out.to_csv(os.path.join(Config.RESULT_SAVE_DIR, f"pred.csv"), index=False)





