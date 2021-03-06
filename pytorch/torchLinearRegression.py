import torch
import subprocess
from torchvision.transforms import transforms
import numpy as np
import os
from torch import nn as nn


# hyperparameter setup
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# define model
linear = nn.Linear(in_features=1, out_features=1)
transform = transforms.ToTensor()
# define the optimizer and loss function
criterion = nn.MSELoss()
opt = torch.optim.SGD(linear.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    y_pre = linear(transform(x_train))
    loss = criterion(y_pre, transform(y_train))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch%10 == 0:
        print(f"epoch={epoch}, loss={loss.item()}")
        print(f'weight.grad={linear.weight.grad}, bias.grad={linear.bias.grad}')
        print(f'weight={linear.weight}, bias={linear.bias}')















if __name__ == '__main__':
    pass
