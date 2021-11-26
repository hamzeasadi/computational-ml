# import required modules
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn as nn
from torchvision.transforms import transforms


# set hyper parameters
num_classes = 10
num_epochs = 1
learning_rate = 1e-3
batch_size = 100
input_shape = (1, 28, 28)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = os.path.join(os.getcwd(), 'data/checkpoint', 'checkpoint.pt')
best_model_path = os.path.join(os.getcwd(), 'data/best_model' 'best_model.pt')
