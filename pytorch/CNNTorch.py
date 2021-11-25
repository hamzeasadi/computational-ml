# import required modules
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn as nn
import torchvision
from torchvision.transforms import transforms

# load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data/', train=True, download=True,
                                            transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='data/', train=False, download=True,
                                            transform=transforms.ToTensor())

                                                                                        












if __name__ == '__main__':
    pass
