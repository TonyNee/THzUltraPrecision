"""
=============================================================================
模块名称: model.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统
作　　者: TonyNee
创建日期: 2025-12-18
最后修改: 2026-05-27
=============================================================================

功能概述:
  本模块定义了项目的所有神经网络模型, 并提供基于装饰器的模型注册表机制。
  通过 MODEL_REGISTRY 字典和 @register_model 装饰器, 实现模型的可插拔架构,
  使得 Config 类只需指定 MODEL_TYPE 字符串即可切换不同模型。

包含模型:
  1. ResMLP           — 残差多层感知机 (128-256-256-128-64-1 with SiLU)
                        采用残差学习范式: 输出 = 输入 + 网络校正量 (ΔF)
  2. BpnnPaper        — 论文复现 BPNN 网络 (500-50-10 with tanh + purelin)
                        严格遵循参考论文的层数和激活函数选择
  3. BPNN (demo)      — 简化 BPNN 网络 (64-128-64-1 with ReLU)
                        用于快速验证和演示
  4. LinearRegression — 单层线性回归模型 (1-1)
                        作为最简基线, 验证非线性模型的相对增益

架构说明:
  - 每个模型类内嵌 DEFAULT_CONFIG 字典, 提供推荐的训练超参数
  - 使用 @register_model("name") 装饰器将模型注册到全局 MODEL_REGISTRY
  - ResMLP 的残差连接设计使得网络学习频率修正量, 而非绝对频率值,
    这有利于降低学习难度、提高校准精度
=============================================================================
"""

import torch
import torch.nn as nn

# =============================================================================
# 全局模型注册表
# =============================================================================
# 键为小写模型名称字符串, 值为 nn.Module 子类
# 由 @register_model 装饰器自动填充
MODEL_REGISTRY = {}


def register_model(name):
    """
    模型注册装饰器

    将模型类以给定名称注册到全局 MODEL_REGISTRY 字典中。
    名称统一转为小写, 重复注册会引发 KeyError。

    参数:
      name: 模型注册名称 (不区分大小写)

    返回:
      decorator: 类装饰器函数

    使用示例:
      @register_model("resmlp")
      class ResMLP(nn.Module):
          ...
    """
    def decorator(cls):
        key = name.lower()
        if key in MODEL_REGISTRY:
            raise KeyError(f"Model '{key}' already registered")
        MODEL_REGISTRY[key] = cls
        return cls
    return decorator


# =============================================================================
# 模型 1: ResMLP — 残差多层感知机
# =============================================================================
# 核心思想: 网络不直接预测绝对频率, 而是预测输入频率与真实频率之间的
# 残差修正量 ΔF = F_true - F_measured, 最终输出 = x + ΔF。
# 这种残差学习范式使得网络只需学习一个小量修正, 显著降低优化难度。
# =============================================================================
@register_model("resmlp")
class ResMLP(nn.Module):
    """
    高精度频率校准残差网络

    网络结构:
      Input(1) -> Linear(128) -> SiLU -> Linear(256) -> SiLU -> Linear(256)
               -> SiLU -> Linear(128) -> SiLU -> Linear(64) -> SiLU -> Linear(1) -> Output(1)

    残差连接:
      Output = Input + net(Input), 即网络学习的是频率修正量 ΔF

    默认超参数:
      batch_size=32, lr=2e-5, epochs=10000, patience=500
      损失函数: MSE, 优化器: Adam, 调度器: CosineAnnealingLR
    """

    # 模型默认训练配置
    DEFAULT_CONFIG = {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "epochs": 10000,
        "patience": 500,
        "loss": "MSE",
        "optimizer": "Adam",
        "scheduler": {
            "type": "CosineAnnealingLR",
            "T_max": "epochs"                                   # 占位符, 由 Config.init() 替换为实际值
        }
    }

    def __init__(self):
        """初始化 6 层残差 MLP, 隐藏层使用 SiLU (Sigmoid Linear Unit) 激活"""
        super().__init__()

        # 全连接序列: 1 -> 128 -> 256 -> 256 -> 128 -> 64 -> 1
        # 全部使用 SiLU 激活 (也称 Swish, 平滑非单调, 在深层网络中优于 ReLU)
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
        # 备选: GELU 激活版本 (保留供消融实验)
        # self.net = nn.Sequential(
        #     nn.Linear(1, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 256),
        #     nn.GELU(),
        #     nn.Linear(256, 256),
        #     nn.GELU(),
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 64),
        #     nn.GELU(),
        #     nn.Linear(64, 1)
        # )

    def forward(self, x):
        """
        前向传播 — 残差学习

        参数:
          x: 输入张量, shape (batch, 1), 为测量的太赫兹频率 (GHz)

        返回:
          tensor: 校正后的频率, shape (batch, 1)
                  计算式为 x + correction,
                  其中 correction = net(x) 即网络学习的频率修正量
        """
        correction = self.net(x)                                # 网络预测的修正量 ΔF
        return x + correction                                   # 残差连接: 输入 + 修正量


# =============================================================================
# 模型 2: BpnnPaper — 论文 BPNN 网络
# =============================================================================
# 严格遵循参考论文的 BPNN 结构:
#   - 三层隐藏层: 500-50-10
#   - 第一隐藏层用 tanh (双曲正切), 后续层用 purelin (线性/恒等)
#   - 优化器: Rprop (弹性反向传播), lr=0.035
# =============================================================================
@register_model("bpnn")
class BpnnPaper(nn.Module):
    """
    论文复现 BPNN 校准网络

    网络结构:
      Input(1) -> tanh(Linear 500) -> Linear 50 -> Linear 10 -> Linear 1 -> Output

    默认超参数:
      batch_size=64, lr=0.035, epochs=100000, patience=5000
      损失函数: Huber, 优化器: Rprop, 无学习率调度器
    """

    DEFAULT_CONFIG = {
        "batch_size": 64,
        "learning_rate": 0.035,                                 # Rprop 推荐学习率
        "epochs": 100000,
        "patience": 5000,
        "loss": "Huber",                                        # Huber 损失对异常值更鲁棒
        "optimizer": "Rprop",                                   # 弹性反向传播, 论文指定
        "scheduler": None,                                      # Rprop 不使用调度器
    }

    def __init__(self):
        """
        初始化论文 BPNN: 4 层全连接

        激活函数方案:
          第一隐藏层 -> tanh (非线性变换, 输出范围 (-1, 1))
          后续层    -> purelin (线性传递, 即无激活函数)
        """
        super().__init__()

        # 4 层全连接: 1 -> 500 -> 50 -> 10 -> 1
        self.hidden1 = nn.Linear(1, 500)                        # 第一隐藏层 (最大容量)
        self.hidden2 = nn.Linear(500, 50)                       # 第二隐藏层 (压缩)
        self.hidden3 = nn.Linear(50, 10)                        # 第三隐藏层 (瓶颈)
        self.output  = nn.Linear(10, 1)                         # 输出层 (标量频率)

    def forward(self, x):
        """
        前向传播 — BPNN 论文结构

        激活顺序: tanh -> purelin -> purelin -> linear

        参数:
          x: 输入张量, shape (batch, 1)

        返回:
          tensor: 预测频率值, shape (batch, 1)
        """
        x = torch.tanh(self.hidden1(x))                         # 第一层: tanh 非线性激活
        x = self.hidden2(x)                                     # 第二层: purelin (线性)
        x = self.hidden3(x)                                     # 第三层: purelin (线性)
        x = self.output(x)                                      # 输出层: 线性
        return x


# =============================================================================
# 模型 3: BPNN (demo) — 简化演示网络
# =============================================================================
@register_model("demo")
class BPNN(nn.Module):
    """
    简化 BPNN 演示网络

    网络结构:
      Input(1) -> ReLU(Linear 64) -> ReLU(Linear 128) -> ReLU(Linear 64) -> Linear 1

    用途: 快速验证训练流程, 网络浅、参数少、收敛快
    """

    DEFAULT_CONFIG = {
        "batch_size": 64,
        "learning_rate": 0.035,
        "epochs": 100000,
        "patience": 2000,
        "loss": "Huber",
        "optimizer": "Rprop",
        "scheduler": {
            "type": "CosineAnnealingLR",
            "T_max": "epochs"
        }
    }

    def __init__(self):
        """初始化简化 4 层 BPNN, 全部使用 ReLU 激活"""
        super(BPNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)                                    # 输出层 (无激活, 回归任务)
        )

    def forward(self, x):
        """
        前向传播

        参数:
          x: 输入张量, shape (batch, 1)

        返回:
          tensor: 预测频率, shape (batch, 1)
        """
        return self.model(x)


# =============================================================================
# 模型 4: LinearRegression — 线性基线模型
# =============================================================================
# 目的: 作为最简基线, 验证非线性模型的相对增益。
# 若非线性模型的指标与线性回归无明显差异, 则说明问题本身接近线性,
# 无需复杂模型; 若指标显著优于线性回归, 则验证了非线性建模的必要性。
# =============================================================================
@register_model("linear")
class LinearRegression(nn.Module):
    """
    线性回归基线模型

    网络结构:
      Input(1) -> Linear(1, 1) -> Output(1)

    实质就是 y = wx + b, 作为所有非线性模型的对齐基线
    """

    DEFAULT_CONFIG = {
        "batch_size": 64,
        "learning_rate": 0.01,
        "epochs": 100000,
        "patience": 20000,
        "loss": "MSE",
        "optimizer": "Adam",
        "scheduler": None,
    }

    def __init__(self):
        """初始化单层线性变换 (1 输入, 1 输出)"""
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        """
        前向传播 — 线性变换 y = Wx + b

        参数:
          x: 输入张量, shape (batch, 1)

        返回:
          tensor: 线性预测值, shape (batch, 1)
        """
        return self.linear(x)
