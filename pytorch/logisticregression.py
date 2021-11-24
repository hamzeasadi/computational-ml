import torch
import numpy as np
from matplotlib import pyplot as plt
import torchvision
from torchvision.transforms import transforms
import os
from torch import nn as nn

train_dataset = torchvision.datasets.MNIST(root='data/', transform=transforms.ToTensor(), train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())












if __name__ == '__main__':
    pass
