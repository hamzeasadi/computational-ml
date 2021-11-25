import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn as nn
from torchvision.transforms import transforms
import torchvision

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# load train and test dataset
train_dataset = torchvision.datasets.MNIST(root='data/', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='data/', train=False, download=True)









if __name__ == '__main__':
    pass
