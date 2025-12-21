import torch
import torch.nn as nn

class BPNN(nn.Module):
    def __init__(self):
        super(BPNN, self).__init__()
        # 三个隐藏层：500, 50, 10个神经元
        self.hidden1 = nn.Linear(1, 500)  # 输入层到第一隐藏层
        self.hidden2 = nn.Linear(500, 50) # 第一隐藏层到第二隐藏层
        self.hidden3 = nn.Linear(50, 10)  # 第二隐藏层到第三隐藏层
        self.output = nn.Linear(10, 1)    # 第三隐藏层到输出层
        
        # 激活函数：tan-sigmoid, purelin, purelin
        # tan-sigmoid 对应 torch.tanh
        # purelin 对应恒等映射（无激活函数）
        
    def forward(self, x):
        # 第一隐藏层：tan-sigmoid
        x = torch.tanh(self.hidden1(x))
        # 第二隐藏层：purelin (无激活函数)
        x = self.hidden2(x)
        # 第三隐藏层：purelin (无激活函数)
        x = self.hidden3(x)
        # 输出层
        x = self.output(x)
        return x

# 可选：创建一个函数来实例化模型
def create_model(device=None):
    model = BPNN()
    if device:
        model = model.to(device)
    return model

# 可选：创建一个函数来获取优化器和损失函数
def get_training_components(model):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.035) # 根据论文设置学习率为0.035
    return criterion, optimizer

# 可选：添加训练配置函数
def get_training_config():
    config = {
        'expected_error': 1e-6,      # 期望训练误差 10^-6
        'max_iterations': 1000000,   # 最大迭代次数 10^6
        'learning_rate': 0.035,      # 学习率
    }
    return config

