# import required packages and librarie
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn as nn
from torchvision.transforms import transforms


# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load datasets
train_dataset = torchvision.datasets.MNIST(root='data/', train=True, download=True,
                                        transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='data/', train=False, download=True,
                                            transform=transforms.ToTensor())

# create torch dataloader util
train_dl = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                        shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

                                        


















if __name__ == '__main__':
    pass
