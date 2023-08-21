import torch.nn as nn
import torch

class FC(nn.Module):
    def __init__(self, input_dim):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.norm_1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.norm_2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        y = torch.relu(self.fc1(x))
        y = self.norm_1(y)
        y = torch.relu(self.fc2(y))
        y = self.norm_2(y)
        y = torch.softmax(self.fc3(y), dim = 1)
        return y