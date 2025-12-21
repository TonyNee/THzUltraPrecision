import torch
import torch.nn as nn

class BPNN(nn.Module):
    def __init__(self):
        super(BPNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        return self.model(x)

# 可选：创建一个函数来实例化模型
def create_model(device=None):
    model = BPNN()
    if device:
        model = model.to(device)
    return model

# 可选：创建一个函数来获取优化器和损失函数
def get_training_components(model):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)
    return criterion, optimizer
