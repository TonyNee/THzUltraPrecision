import os

class Config:
    # 数据路径配置
    TRAIN_CSV = './data/THz_train_20250928.csv'
    EVAL_CSV = './data/THz_eval_20250928.csv'
    
    # 模型保存路径
    MODEL_SAVE_PATH = './model/resmlp_calibration_best.pth'
    RESULT_SAVE_DIR = "./result_resmlp/"
    
    # 训练超参数
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 2000
    PATIENCE = 200
    
    # 设备配置
    DEVICE = "cuda" if os.environ.get("DEVICE") == "cuda" else "cpu"
    
    # 评估配置
    @classmethod
    def setup_directories(cls):
        """创建必要的目录"""
        os.makedirs(os.path.dirname(cls.MODEL_SAVE_PATH) if os.path.dirname(cls.MODEL_SAVE_PATH) else ".", exist_ok=True)
        os.makedirs(cls.RESULT_SAVE_DIR, exist_ok=True)


