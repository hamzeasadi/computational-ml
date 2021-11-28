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
from IPython.display import display, Image
import wandb



# define pathes
data_path = os.path.join(os.getcwd(), 'data')
checkpoint_name = f"checkpoint-custome-{0}.pt"
model_name = f"best-model-custome-{0}.pt"
checkpoint_path = os.path.join(data_path, 'checkpoint', checkpoint_name)
best_model_path = os.path.join(data_path, 'best_model', model_name)

# define hyper-parameters
num_epochs = 2
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
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
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

# load and transfer model
train_and_val_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True,
                                                          download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False,
                                                download=True, transform=transforms.ToTensor())
# create dataloader
train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_and_val_dataset, lengths=[50000, 10000])
train_dl = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,
                                        batch_size=batch_size)
val_dl = torch.utils.data.DataLoader(dataset=val_dataset, shuffle=True,
                                        batch_size=batch_size)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True,
                                        batch_size=batch_size)

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
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2,
                    padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
        nn.Linear(in_features=7*7*32, out_features=256),
        nn.ReLU()
        )
        self.outlayer = nn.Sequential(
        nn.Linear(in_features=256, out_features=num_classes),
        nn.Softmax()
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.flatten(x)
        x = self.fc(x)
        out = self.outlayer(x)
        return out


model = CustomeCNNFashionMNIST(num_classes=num_cls)
model = model.to(device)

# define model criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


# define and implement trai function
def train(model, opt, criterion, train_data, val_data, epochs, ckp_path, bst_mdl_path, min_val_error_in, is_best):
    """train function"""
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for btch_idx, (images, labels) in enumerate(train_data):
            images = images.to(device)
            labels = labels.to(device)
            pre_cls = model(images)
            loss = criterion(pre_cls, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += (loss.item() - train_loss)/len(labels)

        model.eval()
        with torch.no_grad():
            for btch_idx, (images, labels) in enumerate(val_data):
                images = images.to(device)
                labels = labels.to(device)
                pre_cls = model(images)
                loss = criterion(pre_cls, labels)
                val_loss += (loss.item() - val_loss)/len(labels)

        print(f"epoch={epoch}, train_loss={train_loss}, validation_loss{val_loss}")

        checkpoint_state = {
        'epoch': epoch+1,
        'min_val_error': val_loss,
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict()
        }

        if val_loss > min_val_error_in:
            is_best=True
            min_val_error_in = val_loss
        else:
            is_best = False

        save_ckp(state=checkpoint_state, ckp_path=checkpoint_path, is_best_model=is_best,
                bst_model_path=best_model_path)

    return model



# train(model=model, opt=optimizer, criterion=criterion, train_data=train_dl,
#         val_data=val_dl, epochs=num_epochs, ckp_path=checkpoint_path,
#         bst_mdl_path=best_model_path, min_val_error_in=np.inf, is_best=False)


new_model = CustomeCNNFashionMNIST(num_classes=num_cls)
model, optimizer, epoch, min_val_error = load_ckp(ckp_path=checkpoint_path, model=new_model, optimizer=optimizer)
print(model)




if __name__ == '__main__':
    pass
