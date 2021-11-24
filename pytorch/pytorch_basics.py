import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn as nn
import torchvision
from torchvision.transforms import transforms


# basic autograd
x = torch.tensor(np.random.randn(1), dtype=torch.float32)
w = torch.tensor(np.random.randn(1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

y = w*x + b
# calculate grade
y.backward()

print(f"w grad={w.grad}, b.grad={b.grad}")








if __name__ == '__main__':
    pass
