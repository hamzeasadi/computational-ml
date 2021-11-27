# import required modules
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch import nn as nn
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import shutil

# define pathes
data_path = os.path.join(os.getcwd(), 'data')
checkpoint_name = f"checkpoint-custome-{0}.pt"
model_name = f"best-model-custome-{0}.pt"
checkpoint_path = os.path.join(data_path, 'checkpoint', checkpoint_name)
best_model_path = os.path.join(data_path, 'best_model', model_name)

# define hyper-parameters
num_epochs = 1
batch_size = 100
learning_rate = 1e-3
num_cls = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')















if __name__ == '__main__':
    pass
