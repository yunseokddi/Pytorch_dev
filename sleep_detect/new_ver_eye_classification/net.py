from torch import nn
import torch.nn.functional as F

class Dc_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 120)
        self.linear2 = nn.Linear(120, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
