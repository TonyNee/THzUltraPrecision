import torch
import onnx
import argparse

from config import Config

# =========================
# 初始化
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--mdir", required=True)
args = parser.parse_args()
Config.load_yaml(args.mdir)
device = torch.device(Config.DEVICE)
print(f"Device: {device}")

# =========================
# 创建模型
# =========================
model = Config.MODEL_CLASS().to(device)

# 加载训练好的权重
model.load_state_dict(
    torch.load(
        Config.MODEL_SAVE_PATH,
        map_location=device
    )
)

# 推理模式
model.eval()

print("Model loaded successfully.")

# =========================
# 构造 dummy input
# =========================
dummy_input = torch.randn(1, 1).to(device)

# 如果以后输入变成：
# [batch, 1, 1024]
# 就改成：
# dummy_input = torch.randn(1,1,1024)

# =========================
# 导出 ONNX
# =========================
onnx_path = "resmlp.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,

    export_params=True,
    opset_version=11,
    do_constant_folding=True,

    input_names=["input"],
    output_names=["output"]
)

print(f"ONNX exported to: {onnx_path}")

# =========================
# 验证 ONNX
# =========================
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print("ONNX model check passed.")
