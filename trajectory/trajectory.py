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
# define a funtion for creating directories
def dirCreation(base_path, dirname):
    path = os.path.join(base_path, dirname)
    # if os.path.exists(path):
    #     return path
    # else:
    #     return os.makedirs(path)
    # try:
    #     os.makedirs(path)
    # except FileExistsError as e:
    #     print(f"{e}")
    #     return path
    os.makedirs(path, exist_ok=True)
    return path


# define pathes
paths = dict(
data_path = os.path.join(os.path.dirname(os.getcwd()), 'data'),
dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'leader100turn.csv'),
checkpoint_name = f"checkpoint-lstm-model-{0}.pt",
model_name = f"best-lstm-model-{0}.pt",
checkpoint_path = os.path.join(data_path, 'checkpoint', checkpoint_name),
best_model_path = os.path.join(data_path, 'best_model', model_name)
)

# define hyper parameters
# input_shape = (batch_size, seq_length, feature_size)
hyper = dict(
input_shape=(100, 4, 3), epochs=100, learning_rate=1e-3, num_cls=3,
min_val_error=np.inf, num_layers=2, hidden_size=12, fully_conn_size=9
)


# define and implement save and load checkpoints
def save_ckp(state, ckp_path, is_best_model, bst_model_path):
    torch.save(state, ckp_path)
    if is_best_model:
        shutil.copyfile(src=ckp_path, dst=bst_model_path)

def load_ckp(ckp_path, model, optimizer):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epoch']
    min_val_error = checkpoint['min_val_error']
    return model, optimizer, epoch, min_val_error

def load_bst_model(bst_model_path, model, optimizer):
        checkpoint = torch.load(bst_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        min_val_error = checkpoint['min_val_error']
        return model, optimizer, epoch, min_val_error


# define a class for data loadin and preprocessing
class DataWrangling():
    """this is a class that takes data path, sample_sizeas,and batch size as input and retun train and test data loader
    args:
        data: the data path
    methods:

    Raises:
        ValueError: if sample_size is not a positive integer
    """
    def __init__(self, data_path, sample_size: int, batch_size):
        if (sample_size > 0 and isinstance(sample_size, int)):
            self.sample_size = sample_size
            self.batch_size = batch_size
            self.data_path = data_path
        else:
            raise ValueError(f"Expected a positive integer for sample_size, Got {sample_size}")

    def __call__(self, new_sample_size, new_batch_size):
        self.sample_size = new_sample_size
        self.batch_size = new_batch_size
        print(f"A new sample size and batch size initiated")

    def loadData(self):
        nonNormalized_data = pd.read_csv(self.data_path)
        NormalizedData = (nonNormalized_data - nonNormalized_data.mean())/nonNormalized_data.std()

        return NormalizedData.values

    def dataSplit(self):
        data = self.loadData()

        X, Y = [], []
        for i in range(len(data)-self.sample_size):
            X.append(data[i:i+self.sample_size])
            Y.append(data[i+self.sample_size])

        return np.asarray(X), np.asarray(Y)


    def preProcess(self, test_size=0.1):
        X_data, Y_data = self.dataSplit()
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_size, shuffle=True)

        train_dataset, test_dataset = [], []
        for i in range(len(y_train)):
            train_dataset.append([X_train[i], y_train[i]])
        for i in range(len(y_test)):
            test_dataset.append([X_test[i], y_test[i]])

        trainLoader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        testLoader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        return trainLoader, testLoader


# define a class to make our lstm model
class LstmModel(nn.Module):
    """ a simple basic lstm model that predict the 3 continous output
        args:
            input_dim: the input shape of data
            hidden_size: hidden size of lstm cell
            fully_conn_size: the fully connected layer size of the network.
            num_output: the number of output for the model

        Returns:
            it will be a blueprint for creating our model
        Raises:
            None
    """
    def __init__(self, input_shape, hidden_size, fully_conn_size, num_outputs):
        super(LstmModel, self).__init__()
        # input_shape = (batch_size, seq_len, feature_size)
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.fully_conn_size = fully_conn_size
        self.num_outputs = num_outputs
        self.lstm = nn.LSTM(input_size=input_shape[-1], hidden_size=hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # fc_size = seq_len*hidden_size
        fc_size = 1 * hidden_size
        self.fc_1 = nn.Linear(in_features=fc_size, out_features=fully_conn_size)
        self.outlayer = nn.Linear(in_features=fully_conn_size, out_features=num_outputs)


    def forward(self, x):
        h0 = torch.randn(self.input_shape)
        c0 = torch.randn(self.input_shape)
        x = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.relu(x)
        out = self.outlayer(x)

        return out

model = LstmModel(input_shape=input_shape, hidden_size=hidden_size, fully_conn_size=fully_conn_size, num_outputs=num_outputs)




def main():
    DataPipleline = DataWrangling(data_path=data_path, sample_size=sample_size, batch_size=batch_size)
    trainDataLoader, testDataLoader = DataPipleline.preProcess()






    # print(X_train_tensors.shape)

    # train(model=model, data=X_train_tensors_final, y_train=y_train_tensors, test_data=X_test_tensors_final,
    #     y_test=y_test_tensors, opt=optimizer, criterion=criterion, epochs=1)



if __name__ == '__main__':
    main()
