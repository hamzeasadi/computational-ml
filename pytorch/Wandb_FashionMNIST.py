# import required modules
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import hiddenlayer as hl
import shutil
import wandb
import torch
from torchvision import datasets, transforms
from torch import nn as nn

# define paths to save and load data and results
data_path = os.path.join(os.getcwd(), 'data')
checkpoint_name = f"Wand-fashionMNIST-checkpoint.pt"
best_model_name = f"wand-fashionMNIST-best-model.pt"
best_model_path = os.path.join(data_path, 'best_model', best_model_name)
checkpoint_path = os.path.join(data_path, 'checkpoint', checkpoint_name)

# define data transforms
transform = transforms.Compose(
[transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)]
)

# download data and transform them
train_val_dataset = datasets.FashionMNIST(root=data_path, train=True, download=True,
                                        transform=transform)
test_dataset = datasets.FashionMNIST(root=data_path, train=False, download=True,
                                        transform=transform)
                                                                                









if __name__=='__main__':
    pass
