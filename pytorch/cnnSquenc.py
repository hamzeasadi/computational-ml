# import required packages and librarie
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn as nn
from torchvision.transforms import transforms
import shutil


# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define and implement a train fuction
# save the model with checkpoints
checkpoint_path = os.path.join(os.getcwd(), 'data', 'checkpoint', 'current_checkpoint.pt')
best_model_path = os.path.join(os.getcwd(), 'data', 'best_model', 'best_model_path.pt')

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(src=f_path, dst=best_fpath)

# define and implement load checkpoint function
def load_ckp(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

# define transforms
mean_std_cal = torch.utils.data.DataLoader(dataset=)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.2860, std=0.3530)])

# load datasets
train_dataset = torchvision.datasets.MNIST(root='data/', train=True, download=True,
                                        transform=transform)
test_dataset = torchvision.datasets.MNIST(root='data/', train=False, download=True,
                                            transform=transform)
train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])



# create torch dataloader util
train_dl = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                        shuffle=True)
val_dl = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size,
                                    shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)


# define and create a model
class CnnModel(nn.Module):
    """a simple cnn for minist dataset"""
    def __init__(self, num_classes):
        super(CnnModel, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, stride=1, padding=2, kernel_size=5),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=7*7*32, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out



model = CnnModel(num_classes=num_classes)
model_path = os.path.join(os.getcwd(), 'data', 'cnnSquenc.ckpt')
# define loss and optimizer associated with our model
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(src=f_path, dst=best_fpath)

def train(epochs, valid_loss_min_in, tdata, vdata, , model, optimizer, criteria, checkpoint_path, best_model_path):
    valid_loss_min = valid_loss_min_in
    T_loss = []
    V_loss = []
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for btch_idx, (images, labels) in enumerate(tdata):
            img = images.to(device)
            labels = labels.to(device)
            y_pre = model(img)
            optimizer.zero_grad()
            loss = criterion(y_pre, labels)
            loss.backward()
            optimizer.step()
            train_loss += (loss.item() - train_loss)/len(labels)

        model.eval()
        with torch.no_grad():
            for btch_idx, (images, labels) in enumerate(vdata):
                img = images.to(device)
                labels = labels.to(device)
                y_pre = model(img)
                loss = criterion(y_pre, labels)
                valid_loss += ((loss.item() - valid_loss)/len(labels))
        T_loss.append(train_loss/len(tdata))
        V_loss.append(valid_loss/len(vdata))
        print(f"epoch = {epoch+1} train loss = {train_loss}   val loss = {valid_loss}")
        # make a checkpoint
        checkpoint = {
        'epoch': epoch+1,
        'valid_loss_min': valid_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
        # save a checkpoint
        save_ckp(state=checkpoint, is_best=False, checkpoint_path=checkpoint_path, best_model_path='')
        
    plt.plot(np.arange(len(T_loss)), T_loss, label='train loss')
    plt.plot(np.arange(len(T_loss)), V_loss, label='valid loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


train()

# def train(data, model, epochs):
#
#     for epoch in range(epochs):
#         loss = 0.0
#         for i, (images, labels) in enumerate(data):
#             y_pre = model(images)
#             loss = criterion(y_pre, labels)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         print(f"epoch={epoch+1}, loss={loss.item()}")
#
#
#
# # train(data=train_dl, model=model, epochs=num_epochs)
#
# # load the trained model
# newmodel = CnnModel(num_classes=num_classes)
# newmodel.load_state_dict(torch.load(model_path))
#
# #  write the test code
# with torch.no_grad():
#     correct = 0.0
#     total = 0.0
#     for i, (images, labels) in enumerate(test_dl):
#         y_pre = model(images)
#         # y_max_pre = torch.argmax(y_pre, dim=1)
#         _, y_max_pre = torch.max(y_pre.data, 1)
#         correct += (y_max_pre == labels).sum()
#         total += labels.size(0)
#         if i%10==0:
#             print(f"y_max_pre={y_max_pre}")
#             print(f"labels={labels}")
#     print(f'accuracy={correct/total}')
#
#
#
#
#
#
#
#
#
#
#
#
# if __name__ == '__main__':
#     pass
