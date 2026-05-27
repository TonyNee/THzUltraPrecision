"""
=============================================================================
模块名称: backup/model_resmlp.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (早期备份版本)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  早期独立版 ResMLP 模型定义 (HighPrecCalibrator)。
  与当前 model.py 中 ResMLP 类的网络结构完全一致,
  但不包含 MODEL_REGISTRY、DEFAULT_CONFIG 等高级机制。

模型结构:
  Input(1) -> Linear(128)+SiLU -> Linear(256)+SiLU -> Linear(256)+SiLU
           -> Linear(128)+SiLU -> Linear(64)+SiLU -> Linear(1) -> Output(1)

残差学习:
  Output = Input + net(Input), 网络学习频率修正量 ΔF

注意: 此文件已废弃, 当前项目使用 model.py 中的 ResMLP 类。
=============================================================================
"""

import torch
import torch.nn as nn


# =============================================================================
# 高精度频率校准残差网络 HighPrecCalibrator
# =============================================================================
class HighPrecCalibrator(nn.Module):
    """
    早期 ResMLP 实现 (已废弃, 功能被 model.py/ResMLP 取代)

    6 层全连接网络, 隐藏层使用 SiLU 激活, 输出层无激活。
    采用残差学习: 网络预测频率修正量 ΔF, 最终输出 = 输入 + ΔF。
    """

    def __init__(self):
        """初始化 1->128->256->256->128->64->1 全连接序列"""
        super().__init__()

        # Sequential 堆叠: 线性层 + SiLU 激活交替
        # SiLU (Sigmoid Linear Unit): f(x) = x * sigmoid(x), 平滑非单调
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        前向传播 — 残差学习

        参数:
          x: 输入张量 (batch, 1), 测量频率

        返回:
          校正后频率 = x + net(x), 其中 net(x) 为频率修正量
        """
        correction = self.net(x)                                  # 网络预测的修正量 ΔF
        return x + correction                                     # 残差连接: 输入 + 修正量
