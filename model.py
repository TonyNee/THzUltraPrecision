import torch
import torch.nn as nn

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        key = name.lower()
        if key in MODEL_REGISTRY:
            raise KeyError(f"Model '{key}' already registered")
        MODEL_REGISTRY[key] = cls
        return cls
    return decorator


# -------------------------------
#   高精度频率校准网络 ResMLP
# -------------------------------
@register_model("resmlp")
class ResMLP(nn.Module):

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
        super().__init__()

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
        # 残差学习：输出的是 ΔF，最后自动修正
        correction = self.net(x)
        return x + correction      # 输出校正后的频率


# -------------------------------
#   论文BPNN网络 BpnnPaper
# -------------------------------
@register_model("bpnn")
class BpnnPaper(nn.Module):

    DEFAULT_CONFIG = {
        "batch_size": 64,
        "learning_rate": 0.035, 
        "epochs": 100000,
        "patience": 5000,
        "loss": "Huber",
        "optimizer": "Rprop",
        "scheduler": None,
    }

    def __init__(self):
        super().__init__()

        self.hidden1 = nn.Linear(1, 500)
        self.hidden2 = nn.Linear(500, 50)
        self.hidden3 = nn.Linear(50, 10)
        self.output  = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))  # tanh
        x = self.hidden2(x)   # purelin
        x = self.hidden3(x)   # purelin
        x = self.output(x)
        return x

@register_model("demo")
class BPNN(nn.Module):

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
        super(BPNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        return self.model(x)