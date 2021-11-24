import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn as nn
import torchvision
from torchvision.transforms import transforms


# # basic autograd
# x = torch.tensor(np.random.randn(1), dtype=torch.float32)
# w = torch.tensor(np.random.randn(1), requires_grad=True)
# b = torch.randn(1, requires_grad=True)
#
# y = w*x + b
# # calculate grade
# y.backward()
#
# print(f"w grad={w.grad}, b.grad={b.grad}")


# autograd example

# x = torch.randn(10, 3)
#
# y = torch.randn(10, 2)
# # print(dir(nn.Linear()))
# linear = nn.Linear(in_features=3, out_features=2)
# print(f"w: {linear.weight}, b: {linear.bias}")
# opt = torch.optim.SGD(linear.parameters(), lr=0.05)
# criterion = nn.MSELoss()
# epochs = 100
# for epoch in range(1, epochs):
#     y_pre = linear(x)
#     loss = criterion(y, y_pre)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     if epoch % 10 ==0:
#         print(f"epoch = {epoch}, loss={loss.item()}")
#         print(f'w.grad: {linear.weight.grad}, b.grad: {linear.bias.grad}')
#
# with torch.no_grad():
#     print(f"w: {linear.weight.numpy()}")
#     print(f"bias: {linear.bias.numpy()}")







if __name__ == '__main__':
    pass
