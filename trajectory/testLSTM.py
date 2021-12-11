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
from torchsummary import summary
import wandb

wandb.login()

# set random seeds to for consistant algorithm performance check
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# define pathes
data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
dataset_path = os.path.join(data_path, 'SBUX.csv')

# define hyper parameters
epochs = 200
learning_rate = 1e-2
num_classes = 2
num_layers = 2
input_size = 5
hidden_size =2
seq_length = 1

# define loadData function
def loadData(data_path):
    df = pd.read_csv(data_path, index_col = 'Date', parse_dates=True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, 5:6]
    # print(f"x-shape = {np.shape(X)}, y-shape = {np.shape(y)}")
    mm = MinMaxScaler()
    ss = StandardScaler()
    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y)

    X_train = X_ss[:200, :]
    X_test = X_ss[200:, :]
    y_train = y_mm[:200, :]
    y_test = y_mm[200:, :]

    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))
    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    X_train_tensors_final = torch.reshape(X_train_tensors, shape=(X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(X_test_tensors, shape=(X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    return X_train_tensors_final, X_test_tensors_final, y_train_tensors, y_test_tensors



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
        output = output[:, -1, :]
        # hn = hn.view(-1, self.hidden_size)
        hn = output.view(-1, self.hidden_size)
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
        # print(pre.shape, y_train.shape)
        loss = criterion(pre, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss = loss.item()/len(y_train)
        model.eval()
        with torch.no_grad():
            pre_ = model(test_data)
            loss_ = criterion(pre_, y_test)
            eval_loss = loss_.item()/len(y_test)

        EvalOverTrain_loss = eval_loss/train_loss
        print(f"epoch = {epoch}, train-loss = {train_loss}, eval_loss = {eval_loss}")
        # wandb log information
        wandb.log(
        {
        'epoch':epoch,
        'train_loss':train_loss,
        'eval_loss': eval_loss,
        'EvalOverTrain_loss': EvalOverTrain_loss
        }
        )

# wandb configuration
wandb.init(name='lstm2LayerTest', project='lstm2layerTest', entity='hamzeasadi')
wandb.Config.lr = learning_rate

model = LSTM1(num_classes=1, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, seq_length=seq_length)
# define model criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# wandb watch model
wandb.watch(model)

def main():
    X_train_tensors_final, X_test_tensors_final, y_train_tensors, y_test_tensors = loadData(data_path=dataset_path)
    train(model=model, data=X_train_tensors_final, y_train=y_train_tensors, test_data=X_test_tensors_final,
    y_test=y_test_tensors, opt=optimizer, criterion=criterion, epochs=epochs)

    # print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
    # print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)
    # print(model)


if __name__ == '__main__':
    main()
