import torch
import torch.distributed as dist
from torchvision import datasets, transforms

import layers.base_layers as bl
from models.models import BaseModelMP
import losses
import optimizers

BS = 256
LR = 0.0001
EPOCHS = 10
device = "cuda"

def trainloop(train_loader, test_loader):
    torch.manual_seed(0)
    model = BaseModelMP(
        
    )


def main():
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BS, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BS, drop_last=True)
    
    trainloop(train_loader, test_loader)
 
    
        
    
if __name__ == "__main__":
    main()