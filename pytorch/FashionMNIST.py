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

# load dataset
train_val_dataset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True,
                                                        transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='data', train=False, download=True,
                                                transform=transform)

# visualize the data
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
#     img, label = test_dataset[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# train validation data split
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [50000, 10000])

# data loader creation
train_dl = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,
                                        batch_size=batch_size)
val_dl = torch.utils.data.DataLoader(dataset=val_dataset, shuffle=True,
                                        batch_size=batch_size)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True,
                                        batch_size=batch_size)

# define and create the model
class FashionMNISTBaseModel(nn.Module):
    """
    This is a basic model with following hyper parameters:
    layer-1: convolution with 1 input channel and 64 output channel, activation function relue, maxpooling, dropout=0.3
    layer-2: convolution with 16 input channel and 32 output channel, activation function relue, maxpooling, dropout=0.3
    layer-3: fully connected with 7*7*32 input features and 256 output features, activation function relue
    layer-4: fully connected with 256 input features and 10 output features, activation function softmax
    """
    def __init__(self, num_class):
        super(FashionMNISTBaseModel, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(p=0.3)
        )
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(p=0.3)
        )
        conv_out_size = 7*7*32
        self.flatten = nn.Flatten()
        self.layer3 = nn.Sequential(
        nn.Linear(in_features=conv_out_size, out_features=256),
        nn.ReLU()
        )
        self.layer4 = nn.Sequential(
        nn.Linear(in_features=256, out_features=num_class),
        nn.Softmax()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.layer3(x)
        out = self.layer4(x)
        return out

model = FashionMNISTBaseModel(num_class=num_classes).to(device)

# define loss and optimization functions
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# define and implement train function
def train(model, criteria, optimizer, train_data, val_data, epochs, ckp_path, bst_model_path, is_best, val_loss_min_in):
    T_loss = []
    V_loss = []
    for epoch in range(epochs):
        train_loss_avg = 0.0
        val_loss_avg = 0.0
        model.train()
        for btch_idx, (images, labels) in enumerate(train_data):
            images = images.to(device)
            labels = labels.to(device)
            cls_pre = model(images)
            optimizer.zero_grad()
            loss = criteria(cls_pre, labels)
            loss.backward()
            optimizer.step()
            train_loss_avg =+ (loss.item() - train_loss_avg)/len(labels)

        model.eval()
        with torch.no_grad():
            for btch_idx, (images, labels) in enumerate(val_data):
                images = images.to(device)
                labels = labels.to(device)
                cls_pre = model(images)
                loss = criteria(cls_pre, labels)
                val_loss_avg += (loss.item() - val_loss_avg)/len(labels)

        print(f"epoch={epoch+1}, train-loss={train_loss_avg/len(train_data)}, val-loss={val_loss_avg/len(val_data)}")
        checkpoint = {
        'epoch': epoch+1,
        'valid_loss_min': val_loss_avg,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
        # save the checkpoint
        save_ckp(state=checkpoint, ckp_path=ckp_path, bst_model_path=bst_model_path, is_best=False)
        if val_loss_avg > val_loss_min_in:
            save_ckp(state=checkpoint, ckp_path=ckp_path, bst_model_path=bst_model_path, is_best=True)
            val_loss_min_in = val_loss_avg
        T_loss.append(train_loss_avg/len(train_data))
        V_loss.append(val_loss_avg/len(val_data))
    np.savetxt(fname=os.path.join(os.getcwd(), 'data', 'T_loss.csv'), X=np.array(T_loss), delimiter=',')
    np.savetxt(fname=os.path.join(os.getcwd(), 'data', 'V_loss.csv'), X=np.array(V_loss), delimiter=',')
    plt.plot(np.arange(len(T_loss)), T_loss, label='train loss')
    plt.plot(np.arange(len(T_loss)), V_loss, label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('avg loss')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'data/images', 'train-val-error.png'))
    plt.show()
    return model


train(model=model, criteria=criterion, optimizer=opt, train_data=train_dl, val_data=val_dl, epochs=num_epochs,
        ckp_path=checkpoint_path, bst_model_path=best_model_path, is_best=False, valid_loss_min=np.inf)


# load checkpoint data
def load_ckp(ckp_path, model, optimizer):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    val_min_error = checkpoint['valid_loss_min']
    epoch = checkpoint['epoch']
    return model, optimizer, epoch, val_min_error

untrain_model = FashionMNISTBaseModel(num_class=num_classes)
model, optimizer, epoch, val_min_error = load_ckp(ckp_path=checkpoint_path, model=untrain_model, optimizer=opt)


if __name__ == '__main__':
    pass
