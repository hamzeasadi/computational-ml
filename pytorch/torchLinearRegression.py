import torch
import subprocess
from torchvision.transforms import transforms
import numpy as np
import os
from torch import nn as nn




git_message = input("Please enter your message for git commit:")
git_dir = '/Users/hamzeasadi/python/computationalML/computational-ml/'
if git_message:
    subprocess.run(['mgit ', git_message], cwd=git_dir)
