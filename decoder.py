import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, 32)
        self.var = nn.Linear(64, 32)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = torch.sigmoid(x)
        return x

