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

# data loader creation
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [50000, 10000])





if __name__ == '__main__':
    pass
