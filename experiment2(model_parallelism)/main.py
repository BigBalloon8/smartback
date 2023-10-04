import os 

import torch
import torch.distributed as dist
from torchvision import datasets, transforms

import layers.custom_layers as cl
from models.models import CustomModelMP
import losses
import optimizers

BS = 256
LR = 0.0001
EPOCHS = 10
device = "cuda"

def trainloop(train_loader, test_loader):
    torch.manual_seed(0)
    model = CustomModelMP(
        
    )

def main():
    #local_rank = int(os.environ["LOCAL_RANK"])
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    print(dataset1[1:3])
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BS, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BS, drop_last=True)
    #trainloop(train_loader, test_loader)
 
    
        
    
if __name__ == "__main__":
    main()