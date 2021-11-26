# import required modules
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn as nn
from torchvision.transforms import transforms
import shutil


# set hyper parameters
num_classes = 10
num_epochs = 1
learning_rate = 1e-3
batch_size = 100
input_shape = (1, 28, 28)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = os.path.join(os.getcwd(), 'data/checkpoint', 'checkpoint.pt')
best_model_path = os.path.join(os.getcwd(), 'data/best_model' 'best_model.pt')

# checkpoint save function
def save_ckp(state, is_best, ckp_path, bst_model_path):
    torch.save(state, ckp_path)
    if is_best:
        shutil.copyfile(src=ckp_path, dst=bst_model_path)

# make transforms for the data preprocess
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.2860, std=0.3530)])


# train_val_dataset = torchvision.datasets.FashionMNIST(root='data', train=True, )






if __name__ == '__main__':
    pass
