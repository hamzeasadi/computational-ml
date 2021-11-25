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
