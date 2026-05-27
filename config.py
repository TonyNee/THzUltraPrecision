"""
=============================================================================
模块名称: config.py
项目名称: THzUltraPrecision - 太赫兹超高精度频率校准系统
作　　者: TonyNee
创建日期: 2025-12-18
最后修改: 2026-05-27
=============================================================================

功能概述:
  本模块是项目的中心配置管理器, 包含两大核心组件:
  1. Config 类  —— 统一管理模型选择、训练超参数、I/O 路径, 并提供 YAML
                   配置文件的保存/加载/更新功能, 以及损失函数、优化器、
                   学习率调度器的工厂方法
  2. Utils 类  —— 可视化工具类, 提供残差图中按频率段标注最大绝对值的方法

使用方式:
  【训练时】:
      from config import Config
      Config.init()           # 初始化模型、路径、超参数
      Config.save_yaml()      # 保存配置文件到输出目录
  【评估时】:
      from config import Config
      Config.load_yaml(...)   # 从输出目录加载已有配置
      Config.update_yaml(...) # 更新评估指标到配置文件

架构说明:
  - 所有配置项均为类属性 (classmethod), 无需实例化即可全局访问
  - 通过 MODEL_TYPE 字符串从 MODEL_REGISTRY 中查找对应的模型类
  - 模型的默认超参数通过 MODEL_CLASS.DEFAULT_CONFIG 字典提供
=============================================================================
"""

import os
import time
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from model import MODEL_REGISTRY


class Config:
    """
    全局训练/评估配置管理器 (类级单例模式)

    所有配置项通过类属性定义, 通过 @classmethod 方法操作。
    支持两种工作流程:
      train 流程: init() -> save_yaml()
      eval  流程: load_yaml() -> (推理) -> update_yaml()

    模型选择通过 MODEL_TYPE 字符串驱动, 配合 MODEL_REGISTRY 实现可插拔模型架构。
    """

    # =========================================================================
    # 模型配置
    # =========================================================================
    MODEL_TYPE = "resmlp"                       # 模型类型标识, 对应 MODEL_REGISTRY 中的 key
    MODEL_ARCH = [128, 256, 256, 128, 64]       # ResMLP 网络隐藏层维度序列
    # MODEL_TYPE = "bpnn"                       # 备选: 论文 BPNN 模型
    # MODEL_ARCH = [500, 50, 10]                # 备选: BPNN 隐藏层维度 (500-50-10)
    MODEL_CLASS = None                          # 运行时由 init()/load_yaml() 从注册表解析

    # =========================================================================
    # 输入数据路径
    # =========================================================================
    TRAIN_CSV = "./input/scale/111/train.csv"   # 训练集 CSV (每行: 测量频率, 真实频率)
    EVAL_CSV  = "./input/scale/111/eval.csv"    # 评估集 CSV (格式同上)

    # =========================================================================
    # 输出路径 (由 init() 根据 RUN_TIME 自动生成, 例如 ./output/resmlp/20251218120000/)
    # =========================================================================
    RUN_TIME = None                             # 运行时间戳, 格式 YYYYMMDDHHMMSS
    MODEL_SAVE_DIR = None                       # 模型保存目录
    RESULT_SAVE_DIR = None                      # 结果保存目录 (曲线图、CSV 等)
    MODEL_SAVE_PATH = None                      # 最终模型权重文件完整路径 (.pth)

    # =========================================================================
    # 训练超参数 (None 表示使用模型的 DEFAULT_CONFIG)
    # =========================================================================
    K_FOLDS = 5                                 # K 折交叉验证折数
    CV_SEED = 42                                # K 折划分的随机种子 (保证可复现)
    BATCH_SIZE = None                           # 批量大小
    LEARNING_RATE = None                        # 学习率
    EPOCHS = None                               # 最大训练轮数
    PATIENCE = None                             # 早停耐心值 (连续 PATIENCE 轮无改善则停止)
    LOSS_TYPE = None                            # 损失函数类型: "Huber" / "MSE" / "MAE"
    OPTIMIZER_TYPE = None                       # 优化器类型: "AdamW" / "Adam" / "Rprop" / "SGD"
    SCHEDULER_CFG = None                        # 学习率调度器配置字典 (含 type 等参数)

    # =========================================================================
    # 评估指标列表
    # =========================================================================
    METRICS = ["MAE", "MSE", "RMSE", "R2"]      # 默认计算的平均绝对误差、均方误差、均方根误差、决定系数

    # =========================================================================
    # 运算设备: 优先 CUDA, 否则 CPU
    # =========================================================================
    DEVICE = "cuda" if os.environ.get("DEVICE") == "cuda" else "cpu"


    # =========================================================================
    # 公开方法 —— 训练/评估主流程调用
    # =========================================================================

    @classmethod
    def init(cls):
        """
        训练流程初始化 (训练脚本调用)

        执行顺序:
          1. 从 MODEL_REGISTRY 查找 MODEL_TYPE 对应的模型类
          2. 解析模型类的 DEFAULT_CONFIG, 填充未显式设置的超参数
          3. 生成时间戳, 创建输出目录结构

        异常:
          ValueError — 当 MODEL_TYPE 不在注册表中时抛出
        """
        # ---- 步骤 1: 根据 MODEL_TYPE 查找注册模型 ----
        model_key = cls.MODEL_TYPE.strip().lower()
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown MODEL_TYPE '{cls.MODEL_TYPE}'. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

        # 将 MODEL_CLASS 绑定到实际的 nn.Module 子类
        model_class = MODEL_REGISTRY[model_key]
        cls.MODEL_TYPE = model_key
        cls.MODEL_CLASS = model_class

        # ---- 步骤 2: 解析模型默认配置 (DEFAULT_CONFIG) ----
        # 优先级: 显式设置的值 > 模型默认值
        default_cfg = getattr(model_class, "DEFAULT_CONFIG", {})
        cls.BATCH_SIZE     = getattr(cls, "BATCH_SIZE", None)     or default_cfg.get("batch_size")
        cls.LEARNING_RATE  = getattr(cls, "LEARNING_RATE", None)  or default_cfg.get("learning_rate")
        cls.EPOCHS         = getattr(cls, "EPOCHS", None)         or default_cfg.get("epochs")
        cls.PATIENCE       = getattr(cls, "PATIENCE", None)       or default_cfg.get("patience")
        cls.LOSS_TYPE = cls.LOSS_TYPE or default_cfg.get("loss")
        cls.OPTIMIZER_TYPE = cls.OPTIMIZER_TYPE or default_cfg.get("optimizer")
        cls.SCHEDULER_CFG = cls.SCHEDULER_CFG or default_cfg.get("scheduler", {})

        # 若调度器 T_max 为 "epochs" 占位符, 替换为实际 EPOCHS 值
        if cls.SCHEDULER_CFG is not None and cls.SCHEDULER_CFG.get("T_max") == "epochs":
            cls.SCHEDULER_CFG = dict(cls.SCHEDULER_CFG)           # 复制一份避免修改原字典
            cls.SCHEDULER_CFG["T_max"] = cls.EPOCHS

        # ---- 步骤 3: 初始化时间戳与输出路径 ----
        cls.RUN_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())
        arch_str = "-".join(map(str, cls.MODEL_ARCH))              # 例如 "128-256-256-128-64"

        # 输出目录结构: ./output/{model_type}/{timestamp}/
        cls.MODEL_SAVE_DIR = f"./output/{cls.MODEL_TYPE}/{cls.RUN_TIME}"
        cls.RESULT_SAVE_DIR = cls.MODEL_SAVE_DIR
        cls.MODEL_SAVE_PATH = (
            f"{cls.MODEL_SAVE_DIR}/"
            f"{cls.MODEL_TYPE}_{arch_str}.pth"                     # 例如 resmlp_128-256-256-128-64.pth
        )

        # 递归创建目录
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_SAVE_DIR, exist_ok=True)

    @classmethod
    def save_yaml(cls, filename="config.yaml"):
        """
        保存当前配置为 YAML 文件 (训练脚本调用)

        将模型信息、训练超参数、数据路径、运行时信息、输出路径
        组织为结构化字典后写入 YAML, 确保实验完全可复现。

        参数:
          filename: 配置文件名, 默认 "config.yaml"
        """
        cfg = {
            "model": {
                "type": cls.MODEL_TYPE,
                "arch": cls.MODEL_ARCH,
            },
            "training": {
                "batch_size": cls.BATCH_SIZE,
                "learning_rate": cls.LEARNING_RATE,
                "epochs": cls.EPOCHS,
                "patience": cls.PATIENCE,
                "loss": cls.LOSS_TYPE,
                "optimizer": cls.OPTIMIZER_TYPE,
                "scheduler": cls.SCHEDULER_CFG,
            },
            "data": {
                "train_csv": cls.TRAIN_CSV,
                "eval_csv": cls.EVAL_CSV,
            },
            "runtime": {
                "run_time": cls.RUN_TIME,
                "device": cls.DEVICE,
            },
            "paths": {
                "model_save_dir": cls.MODEL_SAVE_DIR,
                "result_save_dir": cls.RESULT_SAVE_DIR,
                "model_save_path": cls.MODEL_SAVE_PATH,
            },
        }

        # 写入模型输出目录
        path = os.path.join(cls.MODEL_SAVE_DIR, filename)
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, model_dir="./", filename="config.yaml"):
        """
        从 YAML 文件加载配置 (评估脚本调用)

        将 YAML 中的各个节 (model/training/data/runtime/paths) 反序列化回
        对应的类属性, 并通过 MODEL_TYPE 解析 MODEL_CLASS。

        参数:
          model_dir: 模型输出目录路径
          filename:  配置文件名, 默认 "config.yaml"

        异常:
          FileNotFoundError — 当配置文件不存在时抛出
        """
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config yaml not found: {path}")

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        # ---- 恢复模型配置 ----
        cls.MODEL_TYPE = cfg["model"]["type"]
        cls.MODEL_ARCH = cfg["model"]["arch"]
        model_key = cls.MODEL_TYPE.lower()
        cls.MODEL_CLASS = MODEL_REGISTRY.get(model_key)            # 从注册表还原模型类

        # ---- 恢复训练超参数 ----
        cls.BATCH_SIZE = cfg["training"]["batch_size"]
        cls.LEARNING_RATE = cfg["training"]["learning_rate"]
        cls.EPOCHS = cfg["training"]["epochs"]
        cls.PATIENCE = cfg["training"]["patience"]
        cls.LOSS_TYPE = cfg["training"]["loss"]
        cls.OPTIMIZER_TYPE = cfg["training"]["optimizer"]
        cls.SCHEDULER_CFG = cfg["training"]["scheduler"]

        # ---- 恢复数据路径 ----
        cls.TRAIN_CSV = cfg["data"]["train_csv"]
        cls.EVAL_CSV  = cfg["data"]["eval_csv"]

        # ---- 恢复运行时信息 ----
        cls.RUN_TIME = cfg["runtime"]["run_time"]
        cls.DEVICE = cfg["runtime"]["device"]

        # ---- 恢复输出路径 ----
        cls.MODEL_SAVE_DIR = cfg["paths"]["model_save_dir"]
        cls.RESULT_SAVE_DIR = cfg["paths"]["result_save_dir"]
        cls.MODEL_SAVE_PATH = cfg["paths"]["model_save_path"]

    @classmethod
    def update_yaml(cls, model_dir="./", metrics_dict=None, filename="config.yaml"):
        """
        将评估指标写入已有 YAML 文件 (评估脚本调用)

        采用原子写入策略: 先写临时文件, 再 rename 替换, 避免写入中断导致配置文件损坏。

        参数:
          model_dir:    模型输出目录路径
          metrics_dict: 指标字典, 例如 {"MAE_MHz": 0.5, "RMSE_MHz": 0.8, ...}
          filename:     配置文件名, 默认 "config.yaml"

        异常:
          ValueError          — metrics_dict 为 None 时抛出
          FileNotFoundError   — 配置文件不存在时抛出
        """
        if metrics_dict is None:
            raise ValueError("metrics_dict must not be None")

        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config yaml not found: {path}")

        # 读取已有配置
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        # 在 evaluation 节下追加指标和时间戳
        evaluation = cfg.get("evaluation", {})
        evaluation["metrics"] = metrics_dict
        evaluation["eval_time"] = time.strftime("%Y%m%d%H%M%S", time.localtime())
        cfg["evaluation"] = evaluation

        # 原子写入: tmp -> rename
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())                                    # 强制刷盘
        os.replace(tmp_path, path)                                  # 原子替换

    # =========================================================================
    # 工厂方法 —— 根据配置项构建 PyTorch 组件
    # =========================================================================

    @classmethod
    def build_loss(cls):
        """
        构建损失函数

        根据 LOSS_TYPE 返回对应的 PyTorch Loss 实例:
          "Huber" -> nn.HuberLoss()  (结合 MSE 与 MAE 的优点, 对异常值更鲁棒)
          "MSE"   -> nn.MSELoss()    (均方误差)
          "MAE"   -> nn.L1Loss()     (平均绝对误差, 即 L1 损失)

        返回:
          nn.Module: 损失函数实例

        异常:
          ValueError — 未知损失类型
        """
        if cls.LOSS_TYPE == "Huber":
            return nn.HuberLoss()
        elif cls.LOSS_TYPE == "MSE":
            return nn.MSELoss()
        elif cls.LOSS_TYPE == "MAE":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {cls.LOSS_TYPE}")

    @classmethod
    def build_optimizer(cls, model):
        """
        构建优化器

        根据 OPTIMIZER_TYPE 返回对应的 PyTorch Optimizer 实例:
          "AdamW" -> optim.AdamW   (Adam + 解耦权重衰减, 推荐默认)
          "Adam"  -> optim.Adam    (标准 Adam)
          "Rprop" -> optim.Rprop   (弹性反向传播, 适合 BPNN 论文复现)
          "SGD"   -> optim.SGD     (随机梯度下降)

        参数:
          model: nn.Module 实例, 优化器将绑定其 parameters()

        返回:
          torch.optim.Optimizer: 优化器实例

        异常:
          ValueError — 未知优化器类型
        """
        if cls.OPTIMIZER_TYPE == "AdamW":
            return optim.AdamW(
                model.parameters(),
                lr=cls.LEARNING_RATE
            )
        elif cls.OPTIMIZER_TYPE == "Adam":
            return optim.Adam(
                model.parameters(),
                lr=cls.LEARNING_RATE
            )
        elif cls.OPTIMIZER_TYPE == "Rprop":
            return optim.Rprop(
                model.parameters(),
                lr=cls.LEARNING_RATE
            )
        elif cls.OPTIMIZER_TYPE == "SGD":
            return optim.SGD(
                model.parameters(),
                lr=cls.LEARNING_RATE
            )
        else:
            raise ValueError(f"Unknown optimizer: {cls.OPTIMIZER_TYPE}")

    @classmethod
    def build_scheduler(cls, optimizer):
        """
        构建学习率调度器

        根据 SCHEDULER_CFG 字典配置调度器。当前支持:
          "CosineAnnealingLR" -> CosineAnnealingLR (余弦退火, 平滑衰减至 0)

        参数:
          optimizer: torch.optim.Optimizer 实例

        返回:
          LRScheduler 或 None (SCHEDULER_CFG 为 None 时表示不使用调度器)

        异常:
          ValueError — 未知调度器类型
        """
        if cls.SCHEDULER_CFG is None:
            return None

        sched_type = cls.SCHEDULER_CFG.get("type")

        if sched_type == "CosineAnnealingLR":
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cls.SCHEDULER_CFG["T_max"]                    # 余弦周期长度 (通常等于 EPOCHS)
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")


class Utils:
    """
    可视化工具类

    提供评估绘图时的辅助功能, 如残差图中按频率范围标注最大绝对误差点。
    当前 THz 系统有两个主要通道: 低频段 (<747 GHz) 和高频段 (>747 GHz)。
    """

    @classmethod
    def annotate_max_abs_by_range(cls, x_data, y_data, label, color_index=0):
        """
        在残差图中标注各频率段内最大绝对误差的位置

        将数据按 747 GHz 阈值分割为低频段和高频段, 分别找出各段内
        绝对值最大的残差点, 用星号标记并附加频率和误差文本标注。

        参数:
          x_data:      频率轴数据 (GHz)
          y_data:      残差轴数据 (MHz)
          label:       标注标签 (如 "MEAS", "PRED")
          color_index: 颜色索引 (0=蓝色, 1=橙色, 对应 matplotlib 默认色板)

        返回:
          dict: 包含 'low' 和 'high' 键, 每个键映射为 (max_abs, max_val, max_freq) 元组
                若无对应段数据则键缺失
        """
        boundary_freq = 747                                        # THz 系统通道分界频率 (GHz)
        colors = ['#1f77b4', '#ff7f0e']                           # matplotlib tab10 颜色
        color = colors[color_index]

        # 将输入转换为 numpy 数组, 便于布尔索引操作
        x_array = np.array(x_data)
        y_array = np.array(y_data)

        # 按 747 GHz 分界, 严格小于/大于, 排除恰好等于边界值的点
        mask_low = x_array < boundary_freq
        mask_high = x_array > boundary_freq

        x_low = x_array[mask_low]
        y_low = y_array[mask_low]
        x_high = x_array[mask_high]
        y_high = y_array[mask_high]

        results = {}

        # ---- 低频段 (<747 GHz) 最大绝对值标注 ----
        if len(y_low) > 0:
            abs_y_low = np.abs(y_low)
            max_abs_idx_low = np.argmax(abs_y_low)                  # 最大绝对值位置
            max_abs_val_low = y_low[max_abs_idx_low]                # 带符号的残差值
            max_abs_freq_low = x_low[max_abs_idx_low]               # 对应频率

            # 星号标记最大误差点
            plt.plot(max_abs_freq_low, max_abs_val_low, '*', color=color, markersize=14,
                    markeredgewidth=1, markeredgecolor='black')

            # 文本标注: 显示误差绝对值 (MHz) 和对应频率 (GHz)
            plt.annotate(f'{abs(max_abs_val_low):.2f} MHz\n@ {max_abs_freq_low:.1f} GHz',
                        xy=(max_abs_freq_low, max_abs_val_low),
                        xytext=(-10, 15 if max_abs_val_low >= 0 else -25),
                        textcoords='offset points',
                        ha='right',
                        va='bottom' if max_abs_val_low >= 0 else 'top',
                        fontsize=10,
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            results['low'] = (abs(max_abs_val_low), max_abs_val_low, max_abs_freq_low)

        # ---- 高频段 (>747 GHz) 最大绝对值标注 ----
        if len(y_high) > 0:
            abs_y_high = np.abs(y_high)
            max_abs_idx_high = np.argmax(abs_y_high)
            max_abs_val_high = y_high[max_abs_idx_high]
            max_abs_freq_high = x_high[max_abs_idx_high]

            # 星号标记
            plt.plot(max_abs_freq_high, max_abs_val_high, '*', color=color, markersize=14,
                    markeredgewidth=1, markeredgecolor='black')

            # 文本标注 (高频段偏移方向相反, 避免重叠)
            plt.annotate(f'{abs(max_abs_val_high):.2f} MHz\n@ {max_abs_freq_high:.1f} GHz',
                        xy=(max_abs_freq_high, max_abs_val_high),
                        xytext=(10, 15 if max_abs_val_high >= 0 else -25),
                        textcoords='offset points',
                        ha='left',
                        va='bottom' if max_abs_val_high >= 0 else 'top',
                        fontsize=10,
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            results['high'] = (abs(max_abs_val_high), max_abs_val_high, max_abs_freq_high)

        return results
