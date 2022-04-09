import torch
from torch import nn as nn
import numpy as np
from torch import optim as optim
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import os


# hypers params
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
input_dim = 784
output_dim = 10
epochs = 1
lr = 1e-4
data_path = os.path.join(os.pardir, 'data')


class FCNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=int(input_dim/10))
        self.out = nn.Linear(in_features=int(input_dim/10), out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.out(x)

        return out


def train(model, train_data, test_data, opt, criterion, epochs, dev):
    
    for epoch in range(epochs):
        train_loss = 0
        for idx, (train_x, train_y) in enumerate(train_data):
            X = train_x.to(dev).reshape((-1, 784))
            
            Y = train_y.to(dev)
            yhat = model(X)
            loss = criterion(yhat, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            print(f"loss = {loss.item()}")

        print(f"epoch={epoch} loss: {train_loss}")
            




def main():
    model = FCNN(input_dim=input_dim, output_dim=output_dim).to(dev)

    mytransforms = transforms.ToTensor()
    
    train_dataset = datasets.MNIST(root=data_path, download=True, train=True, transform=mytransforms)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

    test_dataset = datasets.MNIST(root=data_path, download=True, train=False, transform=mytransforms)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(model=model, train_data=train_dataloader, test_data=test_dataloader, opt=optimizer, criterion=loss, dev=dev, epochs=epochs)




if __name__ == '__main__':
    main()


