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
input_shape = (1, 28, 28)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, input_shape, num_classes):
        super(FashionMNISTBase, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(p=0.3)
        )
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(p=0.3)
        )
        self.outlayer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=2592, out_features=num_classes),
        nn.Softmax()
        )
        # self.flatten = nn.Flatten()
        # n_size = 32*7*7
        # self.fc = nn.Linear(in_features=n_size, out_features=num_classes)
        # self.softmax = nn.softmax(num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        out = self.outlayer(x)
        return out



model = FashionMNISTBase(input_shape=input_shape, num_classes=num_classes).to(device)

# set the model loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# define and implement a train function
def train(data, model, epochs):
    loss = 0.0
    for epoch in range(epochs):

        for i, (images, labels) in enumerate(data):
            y_pre = model(images)
            loss = criterion(y_pre, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%50==0:
                print(f"batch={i+1}, loss={loss.item()}")


train(data=train_dl, model=model, epochs=num_epochs)















if __name__ == '__main__':
    pass
