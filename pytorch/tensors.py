from statistics import mean
import torch
import numpy as np







# hyper-parameters

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')









def main():
    my_tensor = torch.tensor([[1,2,3], [6,4,1]], dtype=torch.float32, requires_grad=False, device=dev)
    norm_tensor = my_tensor.normal_(mean=0, std=1)
    
    btch_t1 = torch.rand(size=(100, 10, 15))
    btch_t2 = torch.rand(size=(100, 15, 11))
    out = torch.bmm(btch_t1, btch_t2)
    print(out.shape)




if __name__ == '__main__':

    main()