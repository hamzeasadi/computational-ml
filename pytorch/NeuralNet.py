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
train_dataset = torchvision.datasets.MNIST(root='data/', train=True, download=True,
                                            transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='data/', train=False, download=True,
                                           transform=transforms.ToTensor())

# create test and train dataloader
train_dl = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

# Create Fully connected neural network with one hidden layer
class Net(nn.Module):
    """Fully connected neural network with one hidden layer"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out

model = Net(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# model criterion and optimizer
criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# define a train function
model_path = os.path.join(os.getcwd(), 'data', 'neuralnet.ckpt')
def train(data, model, epochs):
    train_loss = []

    for epoch in range(epochs):
        loss = 0.0
        for i, (image, label) in enumerate(data):
            img = image.reshape(-1, model.fc1.in_features)
            y_pre = model(img)
            loss = criterion(y_pre, label)
            train_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"epoch={epoch}, loss={loss.item()}")
        # print(f"fc2={model.weight}, fc2.grad={model.weight.grad}")

    torch.save(model.state_dict(), model_path)
    plt.plot(np.arange(len(train_loss)), train_loss)



# train(data=train_dl, model=model, epochs=num_epochs)

# define a test function
def test(data, model):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, (image, label) in enumerate(data):
            img = image.reshape(-1, model.fc1.in_features)
            pre = model(img)
            loss = criterion(y_pre, label)
            y_pre_maxes = torch.argmax(y_pre, dim=1)
            correct += (y_pre_maxes == label).sum()
            total += len(label)
            print(f"batch={i+1}, loss={loss.item()}, accuracy={correct/total}")

mymodel = Net()
mymodel.load_state_dict(torch.load(model_path))
test(data=test_dl,model=mymodel)








if __name__ == '__main__':
    pass
