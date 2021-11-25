# import required modules
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn as nn
import torchvision
from torchvision.transforms import transforms


# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data/', train=True, download=True,
                                            transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='data/', train=False, download=True,
                                            transform=transforms.ToTensor())

# wraps datasets in a dataloader
train_dl = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# create an CNN model
# [batch_size, input_channels, input_height, input_width]  [Batch, Channels, Height, Width]
class CNNmodel(nn.Module):
    """This is a simple cnn model"""
    def __init__(self, in_shape, num_classes):
        self.cnn1 = nn.Conv2(in_channels=1, out_channels=16, kernel_size=5, stride-1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(in_features=7*7*32, out_features=num_classes)
        self.flatten = nn.flatten()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxPool(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxPool(x)
        # x = x.reshape(x.size()[0], -1)
        x = self.flatten(x)
        out = self.fc(x)
        return out
        












if __name__ == '__main__':
    pass
