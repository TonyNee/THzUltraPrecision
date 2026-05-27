"""
=============================================================================
模块名称: backup/config.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统 (早期备份版本)
作　　者: TonyNee
创建日期: 2025-09
最后修改: 2026-05-27
=============================================================================

功能概述:
  早期简化版配置类, 使用硬编码路径和超参数。
  仅用于参考历史版本, 当前项目已迁移至根目录 config.py 的完整 Config 类。

与当前版本的主要差异:
  - 无 YAML 支持 (不能持续化配置)
  - 无 MODEL_REGISTRY 机制 (模型硬编码)
  - 无 K-Fold 相关配置
  - 无模型默认配置 DEFAULT_CONFIG 机制

注意: 此文件已废弃, 不参与当前训练/评估流程。
=============================================================================
"""

import os

class Config:
    """
    早期简化版配置类 (已废弃)

    仅保留了数据路径、模型保存路径和基本训练超参数。
    使用方式为简单的属性访问, 无 YAML 序列化和工厂方法。
    """

    # 数据路径配置 (硬编码, 指向早期数据集)
    TRAIN_CSV = './data/THz_train_20250928.csv'                   # 训练集 CSV
    EVAL_CSV = './data/THz_eval_20250928.csv'                     # 评估集 CSV

    # 模型保存路径
    MODEL_SAVE_PATH = './model/resmlp_calibration_best.pth'       # 最佳模型权重保存路径
    RESULT_SAVE_DIR = "./result_resmlp/"                           # 结果输出目录

    # 训练超参数 (简单固定值, 无模型专属默认配置)
    BATCH_SIZE = 64                                                # 批量大小
    LEARNING_RATE = 1e-3                                           # 学习率
    EPOCHS = 2000                                                  # 最大训练轮数
    PATIENCE = 200                                                 # 早停耐心值

    # 设备配置
    DEVICE = "cuda" if os.environ.get("DEVICE") == "cuda" else "cpu"

    # 目录创建
    @classmethod
    def setup_directories(cls):
        """创建模型和结果输出目录 (若不存在)"""
        # 提取模型保存路径的目录部分, 若为空则使用当前目录
        os.makedirs(
            os.path.dirname(cls.MODEL_SAVE_PATH) if os.path.dirname(cls.MODEL_SAVE_PATH) else ".",
            exist_ok=True
        )
        os.makedirs(cls.RESULT_SAVE_DIR, exist_ok=True)
