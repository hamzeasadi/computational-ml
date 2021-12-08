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
