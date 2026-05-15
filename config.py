"""
Author: TonyNee
Date: 2025-12-18
Description: config 文件
Usage:
- 训练时：
    from config import Config
    Config.init_paths()
    Config.save_yaml()
- 评估时：
    from config import Config
    Config.load_yaml()
    Config.update_yaml()
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
    # 模型配置
    MODEL_TYPE = "resmlp"
    MODEL_ARCH = [128, 256, 256, 128, 64]
    # MODEL_TYPE = "bpnn"
    # MODEL_ARCH = [500, 50, 10]
    MODEL_CLASS = None

    # 输入路径
    TRAIN_CSV = "./input/scale/111/train.csv"
    EVAL_CSV  = "./input/scale/111/eval.csv"

    # 输出路径
    RUN_TIME = None
    MODEL_SAVE_DIR = None
    RESULT_SAVE_DIR = None
    MODEL_SAVE_PATH = None

    # 训练超参数
    K_FOLDS = 5
    CV_SEED = 42
    BATCH_SIZE = None
    LEARNING_RATE = None
    EPOCHS = None
    PATIENCE = None
    LOSS_TYPE = None
    OPTIMIZER_TYPE = None
    SCHEDULER_CFG = None

    # 评估指标
    METRICS = ["MAE", "MSE", "RMSE", "R2"]

    DEVICE = "cuda" if os.environ.get("DEVICE") == "cuda" else "cpu"


    #### PUBLIC METHODS ####
    @classmethod
    def init(cls):
        """初始化所有与时间、模型结构、模型默认配置相关的内容（for train）"""
        # 1. 根据 MODEL_TYPE 选择模型 
        model_key = cls.MODEL_TYPE.strip().lower()
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown MODEL_TYPE '{cls.MODEL_TYPE}'. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

        model_class = MODEL_REGISTRY[model_key]
        cls.MODEL_TYPE = model_key
        cls.MODEL_CLASS = model_class

        # 2. 解析模型 DEFAULT_CONFIG 
        default_cfg = getattr(model_class, "DEFAULT_CONFIG", {})

        cls.BATCH_SIZE     = getattr(cls, "BATCH_SIZE", None)     or default_cfg.get("batch_size")
        cls.LEARNING_RATE  = getattr(cls, "LEARNING_RATE", None)  or default_cfg.get("learning_rate")
        cls.EPOCHS         = getattr(cls, "EPOCHS", None)         or default_cfg.get("epochs")
        cls.PATIENCE       = getattr(cls, "PATIENCE", None)       or default_cfg.get("patience")
        cls.LOSS_TYPE = cls.LOSS_TYPE or default_cfg.get("loss")
        cls.OPTIMIZER_TYPE = cls.OPTIMIZER_TYPE or default_cfg.get("optimizer")
        cls.SCHEDULER_CFG = cls.SCHEDULER_CFG or default_cfg.get("scheduler", {})
        if cls.SCHEDULER_CFG is not None and cls.SCHEDULER_CFG.get("T_max") == "epochs":
            cls.SCHEDULER_CFG = dict(cls.SCHEDULER_CFG)
            cls.SCHEDULER_CFG["T_max"] = cls.EPOCHS

        # 3. 初始化时间与路径 
        cls.RUN_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())
        arch_str = "-".join(map(str, cls.MODEL_ARCH))
        cls.MODEL_SAVE_DIR = f"./output/{cls.MODEL_TYPE}/{cls.RUN_TIME}"
        cls.RESULT_SAVE_DIR = cls.MODEL_SAVE_DIR
        cls.MODEL_SAVE_PATH = (
            f"{cls.MODEL_SAVE_DIR}/"
            f"{cls.MODEL_TYPE}_{arch_str}.pth"
        )
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_SAVE_DIR, exist_ok=True)

    @classmethod
    def save_yaml(cls, filename="config.yaml"):
        """保存 YAML 文件（for train）"""
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

        path = os.path.join(cls.MODEL_SAVE_DIR, filename)
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, model_dir="./", filename="config.yaml"):
        """加载 YAML 文件（for eval）"""
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config yaml not found: {path}")

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        # model
        cls.MODEL_TYPE = cfg["model"]["type"]
        cls.MODEL_ARCH = cfg["model"]["arch"]
        model_key = cls.MODEL_TYPE.lower()
        cls.MODEL_CLASS = MODEL_REGISTRY.get(model_key)

        # training
        cls.BATCH_SIZE = cfg["training"]["batch_size"]
        cls.LEARNING_RATE = cfg["training"]["learning_rate"]
        cls.EPOCHS = cfg["training"]["epochs"]
        cls.PATIENCE = cfg["training"]["patience"]
        cls.LOSS_TYPE = cfg["training"]["loss"]
        cls.OPTIMIZER_TYPE = cfg["training"]["optimizer"]
        cls.SCHEDULER_CFG = cfg["training"]["scheduler"]

        # data 
        cls.TRAIN_CSV = cfg["data"]["train_csv"]
        cls.EVAL_CSV  = cfg["data"]["eval_csv"]

        # runtime
        cls.RUN_TIME = cfg["runtime"]["run_time"]
        cls.DEVICE = cfg["runtime"]["device"]

        # paths
        cls.MODEL_SAVE_DIR = cfg["paths"]["model_save_dir"]
        cls.RESULT_SAVE_DIR = cfg["paths"]["result_save_dir"]
        cls.MODEL_SAVE_PATH = cfg["paths"]["model_save_path"]

    @classmethod
    def update_yaml(cls, model_dir="./", metrics_dict=None, filename="config.yaml"):
        """更新 YAML 文件（for eval）"""
        if metrics_dict is None:
            raise ValueError("metrics_dict must not be None")

        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config yaml not found: {path}")

        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        evaluation = cfg.get("evaluation", {})
        evaluation["metrics"] = metrics_dict
        evaluation["eval_time"] = time.strftime("%Y%m%d%H%M%S", time.localtime())
        cfg["evaluation"] = evaluation

        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno()) 
        os.replace(tmp_path, path)

    #### PRIVATE METHODS ####
    @classmethod
    def build_loss(cls):
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
        if cls.SCHEDULER_CFG is None:
            return None

        sched_type = cls.SCHEDULER_CFG.get("type")

        if sched_type == "CosineAnnealingLR":
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cls.SCHEDULER_CFG["T_max"]
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")

class Utils:

    @classmethod
    # 标注两个范围的最大绝对值
    def annotate_max_abs_by_range(cls, x_data, y_data, label, color_index=0):
        boundary_freq = 747
        colors = ['#1f77b4', '#ff7f0e']
        color = colors[color_index]
        
        # 将数据转换为numpy数组
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        # 分割数据为两个范围
        mask_low = x_array < boundary_freq
        mask_high = x_array > boundary_freq
        
        x_low = x_array[mask_low]
        y_low = y_array[mask_low]
        x_high = x_array[mask_high]
        y_high = y_array[mask_high]
        
        results = {}
        
        # 标注低频范围 (<747 GHz) 的最大绝对值
        if len(y_low) > 0:
            abs_y_low = np.abs(y_low)
            max_abs_idx_low = np.argmax(abs_y_low)
            max_abs_val_low = y_low[max_abs_idx_low]
            max_abs_freq_low = x_low[max_abs_idx_low]
            
            # 统一使用星号(*)标记
            plt.plot(max_abs_freq_low, max_abs_val_low, '*', color=color, markersize=14,
                    markeredgewidth=1, markeredgecolor='black')
            
            # 添加标注，包含频率信息
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
        
        # 标注高频范围 (>747 GHz) 的最大绝对值
        if len(y_high) > 0:
            abs_y_high = np.abs(y_high)
            max_abs_idx_high = np.argmax(abs_y_high)
            max_abs_val_high = y_high[max_abs_idx_high]
            max_abs_freq_high = x_high[max_abs_idx_high]
            
            # 统一使用星号(*)标记
            plt.plot(max_abs_freq_high, max_abs_val_high, '*', color=color, markersize=14,
                    markeredgewidth=1, markeredgecolor='black')
            
            # 添加标注，包含频率信息
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