"""
=============================================================================
模块名称: deprecated/script/model_paper.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (论文 BPNN, 已废弃)
作　　者: TonyNee
创建日期: 2025-11
最后修改: 2026-05-27
=============================================================================

功能概述:
  严格复现参考论文中的 BPNN 网络结构。

模型结构:
  Input(1) -> tanh(Linear 500) -> purelin(Linear 50) -> purelin(Linear 10) -> Linear 1 -> Output

激活函数方案:
  - 第一隐藏层: tanh (tan-sigmoid) — 非线性变换
  - 后续层: purelin (线性/恒等) — 无激活函数

辅助函数:
  - create_model(): 创建模型实例
  - get_training_components(): 获取损失函数 (HuberLoss) 和优化器 (Rprop, lr=0.035)
  - get_training_config(): 获取论文推荐训练配置

废弃原因: 功能已合并到 root/model.py 的 BpnnPaper 类中。

注意: 此文件已废弃。
=============================================================================
"""

import torch
import torch.nn as nn


class BPNN(nn.Module):
    """
    论文复现 BPNN (已废弃, 功能被 model.py/BpnnPaper 取代)

    三层隐藏层: 500 -> 50 -> 10
    第一层用 tanh (tan-sigmoid), 后两层用 purelin (恒等映射)
    """

    def __init__(self):
        super(BPNN, self).__init__()

        # 四个全连接层: 输入(1) -> 500 -> 50 -> 10 -> 输出(1)
        self.hidden1 = nn.Linear(1, 500)                          # 输入层到第一隐藏层 (最大容量)
        self.hidden2 = nn.Linear(500, 50)                         # 第一隐藏层到第二隐藏层 (压缩)
        self.hidden3 = nn.Linear(50, 10)                          # 第二隐藏层到第三隐藏层 (瓶颈)
        self.output = nn.Linear(10, 1)                            # 第三隐藏层到输出层

    def forward(self, x):
        """
        前向传播, 激活顺序: tanh -> purelin -> purelin -> linear

        参数:
          x: (batch, 1) 输入

        返回:
          tensor: (batch, 1) 预测频率
        """
        # 第一隐藏层: tan-sigmoid (tanh)
        x = torch.tanh(self.hidden1(x))
        # 第二隐藏层: purelin (无激活函数)
        x = self.hidden2(x)
        # 第三隐藏层: purelin (无激活函数)
        x = self.hidden3(x)
        # 输出层: 线性
        x = self.output(x)
        return x


def create_model(device=None):
    """
    创建论文 BPNN 模型实例

    参数:
      device: torch.device, 若提供则将模型移动到指定设备
    """
    model = BPNN()
    if device:
        model = model.to(device)
    return model


def get_training_components(model):
    """
    获取论文推荐的训练组件

    返回:
      criterion: HuberLoss(delta=1.0)
      optimizer: Rprop(lr=0.035) — 论文指定学习率
    """
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.035)
    return criterion, optimizer


def get_training_config():
    """
    获取论文推荐的训练配置

    返回:
      dict: 包含 expected_error, max_iterations, learning_rate
    """
    config = {
        'expected_error': 1e-6,                                   # 期望训练误差 10^-6
        'max_iterations': 1000000,                                # 最大迭代次数 10^6
        'learning_rate': 0.035,                                   # Rprop 学习率
    }
    return config
