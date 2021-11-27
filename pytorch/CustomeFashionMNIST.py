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
import hiddenlayer as hl


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

# define and implement network model
"""
Pytorch nn.Conv2d layer expects inputs of (batch_size, channels, height, width).
Scheme 1: CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
Scheme 2: CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC
* Conv - BatchNorm - Activation - DropOut - Pool
Conv - DropOut - BatchNorm - Activation - Pool
Resnet Order:
Convolution
Batch Normalization
ReLU activation function
Maxpooling
"""

class CustomeCNNFashionMNIST(nn.Module):
    """
    define a basic u-shape model for fashionMNIST dataset
    """
    def __init__(self, num_classes):
        super(CustomeCNNFashionMNIST, self).__init__()
        # super().__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1,
                    padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=1,
        padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=1,
        padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
        nn.Linear(in_features=7*7*16, out_features=256),
        nn.ReLU()
        )
        self.outlayer = nn.Sequential(
        nn.Linear(in_features=256*2, out_features=num_classes),
        nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        b1 = self.layer2(x)
        b2 = self.layer3(x)
        fc1 = self.fc(b1)
        fc2 = slef.fc(b2)
        x = torch.cat((fc1, fc2), dim=1)
        out = self.outlayer(x)
        return out


model = CustomeCNNFashionMNIST(num_classes=num_cls)
inp = torch.randn(1, 1, 28, 28)
hl_transform = [ hl.transforms.Prune('Constant') ]
graph = hl.build_graph(model, inp, transforms=hl_transform)
graph.theme = hl.graph.THEMES['blue'].copy()
graph











if __name__ == '__main__':
    pass
