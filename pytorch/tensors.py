from statistics import mean
import torch
import numpy as np







# hyper-parameters

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')









def main():
    tensor = torch.rand(size=(3,3))
    print(tensor)
    print(torch.where(tensor>2, 0, 1))



if __name__ == '__main__':

    main()