import torch
import torch.nn as nn

class LSTMCalibNet(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out  # (batch, seq_len, 1)
