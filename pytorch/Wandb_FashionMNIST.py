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

# define and implement checkpoint save/load functions
def save_ckp(state, checkpoint_path, best_model_path, is_best_model):
    torch.save(state, checkpoint_path)
    if is_best_model:
        shutil.copyfile(src=checkpoint_path, dst=best_model_path)

def load_ckp(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    min_val_error = checkpoint['min_val_error']
    model = model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, epoch, min_val_error



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

# define and implement the model
class WandbTestFashion(nn.Module):
    """this is a simple model to test wandb application"""
    def __init__(self, num_cls=10):
        super().__init__()
        self.layer1 = (
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                    stride=2, padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
        nn.Linear(in_features=7*7*16, out_features=200),
        nn.BatchNorm1d(num_features=200),
        nn.ReLU()
        )
        self.outlayer = nn.Sequential(
        nn.Linear(in_features=200, out_features=num_cls),
        nn.Softmax(dim=1)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.flatten(x)
        x = self.fc(x)
        out = self.outlayer(x)
        return out

wandb.init(project='test', entity='hamzeasadi')





if __name__=='__main__':
    pass
