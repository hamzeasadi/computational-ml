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

# define and implement save and load checkpoints
def save_ckp(state, ckp_path, is_best_model, bst_model_path):
    torch.save(state, ckp_path)
    if is_best_model:
        shutil.copyfile(src=ckp_path, dst=bst_model_path)

def load_ckp(ckp_path, model, optimizer):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    min_val_error = checkpoint['min_val_error']
    return model, optimizer, epoch, min_val_error

def load_bst_model(bst_model_path, model, optimizer):
        checkpoint = torch.load(bst_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        min_val_error = checkpoint['min_val_error']
        return model, optimizer, epoch, min_val_error


















if __name__ == '__main__':
    pass
