import torch

import numpy as np







# hyper-parameters

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')









def main():

    print(dev)





if __name__ == '__main__':

    main()