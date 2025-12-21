import torch
import torch.nn as nn

# -------------------------------
#   高精度频率校准网络 ResMLP
# -------------------------------
class HighPrecCalibrator(nn.Module):
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
