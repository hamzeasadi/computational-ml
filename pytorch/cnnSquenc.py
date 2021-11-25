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

# define and create a model
class CnnModel(nn.Module):
    """a simple cnn for minist dataset"""
    def __init__(self, num_classes):
        super(CnnModel, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, stride=1, padding=2, kernel_size=5),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=7*7*32, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out



model = CnnModel(num_classes=num_classes)
model_path = os.path.join(os.getcwd(), 'data', 'cnnSquenc.ckpt')
# define loss and optimizer associated with our model
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# define and implement a train fuction
def train(data, model, epochs):

    for epoch in range(epochs):
        loss = 0.0
        for i, (images, labels) in enumerate(data):
            y_pre = model(images)
            loss = criterion(y_pre, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch={epoch+1}, loss={loss.item()}")

    torch.save(model.state_dict(), model_path)

# train(data=train_dl, model=model, epochs=num_epochs)

# load the trained model
newmodel = CnnModel(num_classes=num_classes)
newmodel.load_state_dict(torch.load(model_path))

#  write the test code
with torch.no_grad():
    correct = 0.0
    total = 0.0
    for i, (images, labels) in enumerate(test_dl):
        y_pre = model(images)
        y_max_pre = torch.argmax(y_pre, dim=1)
        correct += (y_max_pre == labels).sum()
        total += labels.size(0)
        if i%10==0:
            print(f"y_max_pre={y_max_pre}")
            print(f"labels={labels}")
            print(f'accuracy={correct/total}')












if __name__ == '__main__':
    pass