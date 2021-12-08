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

# define hyper parameters
batch_size = 100
epochs = 100
learning_rate = 1e-2
sample_size = 4


# define a class for data loadin and preprocessing
class DataWrangling():
    """this is a class that takes data path, sample_sizeas,and batch size as input and retun train and test data loader
    args:
        data: the data path
    methods:

    Raises:
        ValueError: if sample_size is not a positive integer
    """
    def __init__(self, data_path, sample_size, batch_size=100):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.data_path = data_path

    def __call__(self, new_sample_size, new_batch_size):
        self.sample_size = new_sample_size
        self.batch_size = new_batch_size
        print(f"A new sample size and batch size initiated")

    def loadData(self):
        return pd.read_csv(self.data_path)

    def dataSplit(self):
        data = self.loadData()
        X, Y = [], []
        for sample in range(len(data)-self.sample_size):
            X.append(data[i:i+self.sample_size])
            Y.append(data[i+self.sample_size])
        return np.asarray(X), np.asarray(Y)


    def preProcess(self, test_size=0.1):
        X_data, Y_data = self.dataSplit()
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_size, shuffle=True)
        scale_x = StandardScaler()
        scale_y = StandardScaler()
        X_train = torch.from_numpy(scale_x.fit_transform(X_train))
        y_train = torch.from_numpy(scale_y.fit_transform(y_train))
        X_test = torch.from_numpy(scale_x.transform(X_test))
        y_test = torch.from_numpy(scale_y.transform(y_test))

        return X_train, X_test, y_train, y_test





# define loadData function
data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'leader100turn.csv')
def loadData(dataPath):
    data = pd.read_csv(dataPath)
    return data

# define dataSampling(data, sample_size)
def dataSampling(data, sample_size):
    """this function takes a numpy multidimentional array and return another multidimentional
    array with (n, sample_size)
    args:
        data: multidimentional array
        sample_size: a positive integer
    return:
        newdata: a new multidimentional array with (n, sample_size) shape
    Raises:
        ValueError: if the sample size is not an positive integer
    """
    if sample_size > 0:
        X = []
        Y = []
        for i in range(len(data)-sample_size):
            X.append(data[i:i+sample_size])
            Y.append(data[i+sample_size])
            # print(data[i:i+sample_size])
        return np.asarray(X), np.asarray(Y)
    else:
        raise ValueError(f"Expected a positive integer for sample_size, Got {sample_size}")



def main():
    mydata = loadData(data_path)
    data = mydata.values
    X_data, Y_data = dataSampling(data=data, sample_size=4)
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.1, shuffle=True)
    print(np.shape(X_train), np.shape(X_test),np.shape(y_train), np.shape(y_test))





if __name__ == '__main__':
    main()
