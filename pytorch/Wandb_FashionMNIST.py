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

# define hyper parameters
hyper = dict(
num_epochs=1, batch_size=100, lr=1e-3, num_cls=10,
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)

# define data transforms
transform = transforms.Compose(
[transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)]
)

# download data and transform them
train_val_dataset = datasets.FashionMNIST(root=data_path, train=True, download=True,
                                        transform=transform)
train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset,
                                                            lengths=[50000, 10000])
test_dataset = datasets.FashionMNIST(root=data_path, train=False, download=True,
                                        transform=transform)

# make a data loader for datasets
train_dl = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,
                                        batch_size=hyper['batch_size'])
val_dl = torch.utils.data.DataLoader(dataset=val_dataset, shuffle=True,
                                    batch_size=hyper['batch_size'])
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True,
                                     batch_size=hyper['batch_size'])                                    






if __name__=='__main__':
    pass
