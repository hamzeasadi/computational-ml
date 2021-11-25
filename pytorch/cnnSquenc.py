# import required packages and librarie
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn as nn
from torchvision.transforms import transforms

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load datasets
train_dataset = torchvision.datasets.MNIST(root='data/', train=True, download=True,
                                        transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='data/', train=False, download=True,
                                            transform=transforms.ToTensor())
                                            

















if __name__ == '__main__':
    pass
