# import required modules
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn as nn
from torchvision.transforms import transforms

# set hyper parameters
num_classes = 10
num_epochs = 1
learning_rate = 1e-3
batch_size = 100

# load dataset
train_dataset = torchvision.datasets.FashionMNIST(root='data/', train=True, download=True,
                                                 transform=transforms.ToTensor())
test_dataset = torchvision.datasets.FashionMNIST(root='data/', train=False, download=True,
                                                transform=transforms.ToTensor())

# dataloader creation from train test
train_dl = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                        shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True,
                                        batch_size=batch_size)
                                                                                                                                                                                

# model setting
"""
[(convolution layer1 filters, convolution layer1 kernel size),
(convolution layer2 filters, convolution layer2 kernel size),
(convolution layer1 Dropout-probability, convolution layer2 Dropout-probability)]

[(64,2),(32,2), (0.3,0.3)]
"""

# define the model and model_patth
model_path = os.path.join(os.getcwd(), 'data', 'baseFashionMNIST.ckpt')

class FashionMNISTBase(nn.Module):
    """
    A basic cnn model for classifing fashion mnist dataset
    """
    def __init__(self, input_shape=(1, 28, 28), num_classes):
        super(FashionMNISTBase, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, outchannels=64, kernel_size=(2, 2), stride=1, padding=2),
        nn.ReLU(),
        nn.Dropout(p=0.3)
        )
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=1, padding=2),
        nn.ReLU(),
        nn.Dropout(p=0.3)
        )
        n_size = self._get_conv_output(input_shape)
        self.fc = nn.Linear(in_features=n_size, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.fc(x)
        return out

# define and implement a train function
def train(data, model, epochs):
    pass















if __name__ == '__main__':
    pass
