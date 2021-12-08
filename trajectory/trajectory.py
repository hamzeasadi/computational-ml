# import required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch import nn as nn
import torchvision
from torchvision.transforms import transforms
import os



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
        newdata = []
        for i in range(len(data)-sample_size):
            newdata.append(data[i:sample_size])
        return newdata
    else:
        raise ValueError(f"Expected a positive integer for sample_size, Got {sample_size}")



def main():
    mydata = loadData(data_path)
    print(mydata.describe())
    # X = mydata.iloc[:, :-1]
    # Y = mydata.iloc[:, 5:6]
    # print(Y)
    # print(mydata.head(5))
    # plt.style.use('ggplot')
    # mydata.plot(label='CLOSE', title='Star Bucks Stock Volume')
    sns.pairplot(data=mydata)
    plt.show()



if __name__ == '__main__':
    main()
