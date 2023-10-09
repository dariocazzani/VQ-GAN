import torch
import torch.nn.functional as F

def main():
    x = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]])
    pad = (3,1,0,2)
    new_x = F.pad(x, pad, mode="constant", value=0)
    print(x)
    print(new_x)

if __name__ == "__main__":
    main()
