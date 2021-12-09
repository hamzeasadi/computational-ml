# import required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch import nn as nn
import torchvision
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import random
import shutil
import math
from torch.autograd import Variable



# set random seeds to for consistant algorithm performance check
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# define pathes
data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'SBUX.csv')

# define hyper parameters
epochs = 1
learning_rate = 1e-2
num_classes = 1
num_layers = 2
input_size = 5
hidden_size =2

# define loadData function
def loadData(data_path):
    pass



# define lstm model and implement
class LSTM1(nn.Module):
    """simple lstm"""
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(in_features=hidden_size, out_features=128)
        self.fc = nn.Linear(in_features=128, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.randn(self.num_layers, x.size(0), self.hidden_size))
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

# define and implement train function
def train(model, data, y_train, test_data, y_test, opt, criterion, epochs):
    for epoch in range(epochs):
        train_loss = 0.0
        eval_loss = 0.0
        model.train()
        pre = model(data)
        print(pre.shape, y_train.shape)
        loss = criterion(pre, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss = loss.item()
        model.eval()
        with torch.no_grad():
            pre_ = model(test_data)
            loss_ = criterion(pre_, y_test)
            eval_loss = loss_.item()

        print(f"train-loss = {train_loss}, eval_loss = {eval_loss}")

model = LSTM1(num_classes=1, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, seq_length=1)
# define model criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)













def main():
    pass


if __name__ == '__main__':
    main()
