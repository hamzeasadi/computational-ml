# import required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch import nn as nn
import torchvision
from torchvision.transforms import transforms

# define loadData function
data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'SBUX.csv')
def loadData(dataPath):
    data = pd.read_csv(dataPath)
    return data


def main():


if __name__ == '__main__':
    main()
