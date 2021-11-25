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
# [batch_size, input_channels, input_height, input_width]
class CNNmodel(nn.Module):
    """This is a simple cnn model"""
    def __init__(self, in_shape, num_classes):
        self.cnn1 = nn.Conv2(in_features)












if __name__ == '__main__':
    pass
