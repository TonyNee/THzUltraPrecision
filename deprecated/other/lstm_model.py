"""
=============================================================================
模块名称: deprecated/other/lstm_model.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (LSTM 实验, 已废弃)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  基于 LSTM 的序列频率校准网络定义。
  将频率测量值组织为序列输入, 通过双向 LSTM 提取时序特征, 预测残差修正量。

模型结构:
  LSTM (2层, hidden=128) -> Linear(hidden_dim -> 1) -> Output(seq_len, 1)

废弃原因:
  频率校准问题是点对点映射 (f_measured -> f_true), 不存在时序依赖关系,
  LSTM 的序列建模能力在此任务上并无优势, 且增加了不必要的复杂度。

注意: 此文件已废弃, 当前项目使用 model.py 中的前馈网络 (ResMLP/BpnnPaper)。
=============================================================================
"""

import torch
import torch.nn as nn


class LSTMCalibNet(nn.Module):
    """
    LSTM 频率校准网络 (已废弃)

    将频率测量值序列输入 LSTM, 预测每个时间步的残差修正量。

    参数:
      hidden_dim: LSTM 隐藏层维度, 默认 128
      num_layers: LSTM 层数, 默认 2
      dropout:    LSTM 层间 dropout 比例, 默认 0.0
    """

    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()

        # 双层 LSTM, batch_first=True 表示输入格式为 (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=1,                                         # 每步输入维度 (单个频率值)
            hidden_size=hidden_dim,                               # 隐藏状态维度
            num_layers=num_layers,                                # LSTM 堆叠层数
            batch_first=True,                                     # 输入输出 shape 为 (batch, seq, feature)
            dropout=dropout                                       # 层间 dropout (仅 num_layers > 1 时有效)
        )

        # 全连接输出头: 将 LSTM 隐藏状态映射到标量预测
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        前向传播

        参数:
          x: 输入张量, shape (batch, seq_len, 1)

        返回:
          tensor: 每个时间步的预测值, shape (batch, seq_len, 1)
        """
        # LSTM 输出: out (batch, seq_len, hidden_dim), _ (h_n, c_n)
        out, _ = self.lstm(x)
        # 全连接层: 将 hidden_dim 映射为标量输出
        out = self.fc(out)
        return out                                                # (batch, seq_len, 1)
