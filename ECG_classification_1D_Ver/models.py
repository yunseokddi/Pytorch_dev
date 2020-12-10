import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, device):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        # self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(self.device)

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(self.device)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


class CNN_v1(nn.Module):
    def __init__(self):
        super(CNN_v1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.fc1 = nn.Linear(10 * 12 * 12, 50)
        self.fc2 = nn.Linear(50, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        print(x.size())
        x = x.view(-1, 10 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = CNN_v1()
model.to('cuda')

summary(model, (128,187,1))