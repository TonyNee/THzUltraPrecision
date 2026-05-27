"""
=============================================================================
模块名称: export_onnx.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统
作　　者: TonyNee
创建日期: 2025-12-20
最后修改: 2026-05-27
=============================================================================

功能概述:
  将训练好的 PyTorch 模型导出为 ONNX (Open Neural Network Exchange) 格式,
  用于跨平台部署 (如 C++ inference server、嵌入式设备等)。

ONNX 导出流程:
  1. 从 config.yaml 恢复模型结构
  2. 加载训练好的权重
  3. 构造 dummy input (用于追踪计算图)
  4. 调用 torch.onnx.export 导出
  5. 使用 onnx.checker 验证导出的模型文件完整性

使用方式:
  python export_onnx.py --mdir ./output/resmlp/20251218120000/
  输出文件: resmlp.onnx (当前目录)
=============================================================================
"""

import torch
import onnx
import argparse

from config import Config

# =============================================================================
# 阶段 0: 环境初始化
# =============================================================================
parser = argparse.ArgumentParser(description="PyTorch 模型 -> ONNX 导出脚本")
parser.add_argument("--mdir", required=True,
                    help="模型输出目录路径")
args = parser.parse_args()

Config.load_yaml(args.mdir)
device = torch.device(Config.DEVICE)
print(f"Device: {device}")

# =============================================================================
# 阶段 1: 创建模型并加载权重
# =============================================================================
# 从注册表中根据配置创建模型实例
model = Config.MODEL_CLASS().to(device)

# 加载训练好的权重参数
model.load_state_dict(
    torch.load(
        Config.MODEL_SAVE_PATH,
        map_location=device                                     # 兼容 CPU/CUDA 跨设备加载
    )
)

# 切换到推理模式 (禁用 dropout/batchnorm 等训练专用层)
model.eval()
print("Model loaded successfully.")

# =============================================================================
# 阶段 2: 构造 dummy input (用于图追踪)
# =============================================================================
# 本模型输入为 (batch_size=1, input_dim=1), 即单个频率值
dummy_input = torch.randn(1, 1).to(device)

# 如果将来输入维度变化 (例如增加频谱特征), 修改此处即可:
# dummy_input = torch.randn(1, 1, 1024)   # batch=1, channels=1, freq_bins=1024

# =============================================================================
# 阶段 3: 导出 ONNX
# =============================================================================
onnx_path = "resmlp.onnx"                                       # 输出文件路径

torch.onnx.export(
    model,                                                      # 待导出的 PyTorch 模型
    dummy_input,                                                # 示例输入 (追踪计算图)
    onnx_path,                                                  # 输出路径

    export_params=True,                                         # 将权重参数嵌入 ONNX 文件
    opset_version=11,                                           # ONNX 算子集版本 (兼容性好)
    do_constant_folding=True,                                   # 折叠常量节点以优化推理

    input_names=["input"],                                      # 输入节点名称
    output_names=["output"]                                     # 输出节点名称
)

print(f"ONNX exported to: {onnx_path}")

# =============================================================================
# 阶段 4: 验证 ONNX 模型完整性
# =============================================================================
# 加载导出的 ONNX 文件
onnx_model = onnx.load(onnx_path)

# 使用 ONNX 内置检查器验证模型结构和算子兼容性
onnx.checker.check_model(onnx_model)

print("ONNX model check passed.")
