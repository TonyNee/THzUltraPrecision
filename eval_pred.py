"""
=============================================================================
模块名称: eval_pred.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统
作　　者: TonyNee
创建日期: 2026-01-13
最后修改: 2026-05-27
=============================================================================

功能概述:
  纯推理脚本, 仅加载模型对输入数据进行预测, 不计算误差指标。
  适用于只有测量频率、没有真实标签的场景 (无监督推理)。

与 eval.py 的区别:
  - eval.py: 需要真实标签, 计算完整误差指标并绘制残差图
  - eval_pred.py: 无需真实标签, 仅输出预测值 CSV, 适合生产环境批量推理

使用方式:
  python eval_pred.py --mdir ./output/resmlp/20251218120000/
  输入 CSV 必须包含至少一列 (测量频率), 可无表头或无标签列
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import torch
import argparse

from config import Config


# =============================================================================
# 阶段 0: 环境配置
# =============================================================================
parser = argparse.ArgumentParser(description="THz 频率校准模型纯推理脚本")
parser.add_argument("--mdir", required=True,
                    help="模型输出目录路径, 含 config.yaml 和 .pth 文件")
args = parser.parse_args()

# 从 YAML 恢复训练配置
Config.load_yaml(args.mdir)
device = torch.device(Config.DEVICE)
print(f"Device: {device}")

# =============================================================================
# 阶段 1: 加载输入数据 (无需标签)
# =============================================================================
# header=None 表示 CSV 无表头, 直接按列索引
df = pd.read_csv(Config.EVAL_CSV, header=None)
x_eval = df.iloc[:, 0].values.astype(np.float32)                # 测量频率 (GHz)

# 转为模型输入格式 (batch, 1)
X = torch.tensor(x_eval.reshape(-1, 1)).to(device)

# =============================================================================
# 阶段 2: 加载模型权重
# =============================================================================
model = Config.MODEL_CLASS().to(device)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
model.eval()

# =============================================================================
# 阶段 3: 批量推理
# =============================================================================
with torch.no_grad():                                           # 关闭自动微分
    y_pred = model(X).cpu().numpy()


# =============================================================================
# 阶段 4: 保存预测结果
# =============================================================================
model_name = Config.MODEL_TYPE.replace(" ", "_")

# 输出 CSV: 仅含测试频率和预测频率
df_out = pd.DataFrame({
    "F_test_GHz": x_eval,                                       # 原始测量频率
    "F_pred_GHz": y_pred.flatten(),                             # 模型校准后的频率
})
df_out.to_csv(os.path.join(Config.RESULT_SAVE_DIR, f"pred.csv"), index=False)
