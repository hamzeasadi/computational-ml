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
model_name = 'logisticModel.ckpt'
data_path = os.path.join(os.getcwd(), 'data')
model_path = os.path.join(data_path, model_name)
model = nn.Linear(in_features=input_size, out_features=num_classes)
# define loss function
criterion = nn.CrossEntropyLoss()
# define optimizer
opt = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

def train(dataLoader, model, epochs):
    num_batchs = len(dataLoader)
    for epoch in range(epochs):
        for i, (image, label) in enumerate(dataLoader):
            img = image.reshape(-1, input_size)
            y_pre = model(img)
            loss = criterion(y_pre, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"epoch={epoch}, loss={loss.item()}")
        print(f"weights={model.weight}, bias={model.bias}")
        print(f"w.grad={model.weight.grad}, bias.grad={model.bias.grad}")

    torch.save(model.state_dict(), model_path)

# train(dataLoader=train_dl, model=model, epochs=num_epochs)

with torch.no_grad():
    # model_dict = torch.load_state_dict(model_path)
    model.load_state_dict(torch.load(model_path))
    num_batchs = len(test_dl)
    correct = 0.0
    total = 0.0
    for i, (image, label) in enumerate(test_dl):
        img = image.reshape(-1, input_size)
        y_pre = model(img)
        loss = criterion(y_pre, label)
        predicted = torch.argmax(y_pre, dim=1)
        correct += sum(predicted==label)
        total += len(label)
    print(f"total correct percentage={correct/total}")












if __name__ == '__main__':
    pass
