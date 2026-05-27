"""
=============================================================================
模块名称: deprecated/script/model.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (v1 BPNN, 已废弃)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  v1 版本 BPNN 模型定义 (1-64-32-1 with ReLU)。

模型结构:
  Input(1) -> Linear(64)+ReLU -> Linear(32)+ReLU -> Linear(1) -> Output

辅助函数:
  - create_model(): 创建模型实例
  - get_training_components(): 获取损失函数 (HuberLoss) 和优化器 (Rprop)

废弃原因: 被 root/model.py 中的注册表机制和 ResMLP/BpnnPaper 取代。

注意: 此文件已废弃。
=============================================================================
"""

import torch
import torch.nn as nn


class BPNN(nn.Module):
    """
    v1 版本 BPNN (已废弃)

    简单 3 层全连接网络:
      Input(1) -> ReLU(64) -> ReLU(32) -> Linear(1)
    """

    def __init__(self):
        super(BPNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),                                    # 输入层 -> 隐藏层1
            nn.ReLU(),                                           # ReLU 激活
            nn.Linear(64, 32),                                   # 隐藏层1 -> 隐藏层2
            nn.ReLU(),                                           # ReLU 激活
            nn.Linear(32, 1)                                     # 隐藏层2 -> 输出层
        )

    def forward(self, x):
        """前向传播: (batch, 1) -> (batch, 1)"""
        return self.model(x)


def create_model(device=None):
    """
    创建 BPNN 模型实例

    参数:
      device: torch.device, 若提供则将模型移动到指定设备

    返回:
      BPNN 实例
    """
    model = BPNN()
    if device:
        model = model.to(device)
    return model


def get_training_components(model):
    """
    获取默认的损失函数和优化器

    参数:
      model: BPNN 实例

    返回:
      (criterion, optimizer): HuberLoss(delta=1.0), Rprop(lr=0.01)
    """
    criterion = nn.HuberLoss(delta=1.0)                          # delta=1.0: 1 GHz以内用MSE
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)   # 弹性反向传播
    return criterion, optimizer
