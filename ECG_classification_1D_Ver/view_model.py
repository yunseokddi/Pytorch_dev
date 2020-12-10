import torch
import numpy as np

from torchsummary import summary
from models import LSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu = 0
num_epochs = 100
stopping_epoch = 20
learning_rate = 1.0

input_size = 1
hidden_size = 128
num_layers = 1
batch_size = 128

num_classes = 5
out = "./result"
model = LSTM(num_classes, input_size, hidden_size, num_layers, device)
model.to(device)

summary(model,(1,128,128))