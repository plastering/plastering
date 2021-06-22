import torch
import torch.nn as nn
from torch.autograd import Variable

# model-stn

class STN(nn.Module):
    def __init__(self, dropout_rate, input_channels):
        super(STN, self).__init__()
        self.d = dropout_rate

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=256, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=384, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=384, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
        )
        self.dropout1 = nn.Dropout(self.d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        norm = x.norm(dim=1, p=2, keepdim=True)
        x = x.div(norm.expand_as(x))

        return x
