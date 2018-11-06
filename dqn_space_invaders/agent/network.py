import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, action_size):
        super(CNN, self).__init__()
        self.conv_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear_1 = nn.Linear(4480, 784)
        self.linear_2 = nn.Linear(784, action_size)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.linear_1(x.view(x.size(0), -1)))
        # not add softmax at the last layer.
        x = self.linear_2(x)
        return x
