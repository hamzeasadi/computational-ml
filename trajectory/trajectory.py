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


# set random seeds to for consistant algorithm performance check
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# define pathes
data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'leader100turn.csv')
checkpoint_name = f"checkpoint-lstm-model-{0}.pt"
model_name = f"best-lstm-model-{0}.pt"
checkpoint_path = os.path.join(data_path, 'checkpoint', checkpoint_name)
best_model_path = os.path.join(data_path, 'best_model', model_name)

# define hyper parameters
batch_size = 100
epochs = 100
learning_rate = 1e-2
sample_size = 4
num_outputs = 3

# define and implement save and load checkpoints
def save_ckp(state, ckp_path, is_best_model, bst_model_path):
    torch.save(state, ckp_path)
    if is_best_model:
        shutil.copyfile(src=ckp_path, dst=bst_model_path)




# define a class for data loadin and preprocessing
class DataWrangling():
    """this is a class that takes data path, sample_sizeas,and batch size as input and retun train and test data loader
    args:
        data: the data path
    methods:

    Raises:
        ValueError: if sample_size is not a positive integer
    """
    def __init__(self, data_path, sample_size: int, batch_size=100):
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
        # X_train = torch.from_numpy(X_train)
        # y_train = torch.from_numpy(y_train)
        # X_test = torch.from_numpy(X_test)
        # y_test = torch.from_numpy(y_test)

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


    def forward(self, x):
        pass




def main():
    DataPipleline = DataWrangling(data_path=data_path, sample_size=sample_size, batch_size=batch_size)
    trainDataLoader, testDataLoader = DataPipleline.preProcess()






if __name__ == '__main__':
    main()
