from torch import nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, in_size, hid_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hid_size,
            batch_first=True,
            num_layers=1
            # proj_size=1,
        )
        self.proj = nn.Linear(in_features=hid_size, out_features=1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.proj(x)
        return x