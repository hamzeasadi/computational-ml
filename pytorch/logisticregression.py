import torch
import numpy as np
from matplotlib import pyplot as plt
import torchvision
from torchvision.transforms import transforms
import os
from torch import nn as nn

# Hyper-parameters
input_size = 28 * 28    # 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='data/', transform=transforms.ToTensor(), train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())

train_dl = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)
# create a logistic model
model = nn.Linear(in_features=input_size, out_features=num_classes)
# define loss function
criterion = nn.CrossEntropyLoss()
# define optimizer
opt = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# CrossEntropyLoss











if __name__ == '__main__':
    pass
