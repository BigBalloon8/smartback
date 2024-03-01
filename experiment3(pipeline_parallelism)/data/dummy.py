import random

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset
import torch.distributed as dist


class DummyDataInt(Dataset):
    def __init__(self, length, shape, vocab_size, chimera=False):
        self.length = length
        self.shape = shape
        self.vocab_size = vocab_size
        self.chimera = chimera
    
    def __len__(self):
        if self.chimera:
            return self.length // 2
        else:
            return self.length
    
    def __getitem__(self, idx):
        if  self.chimera:
            if dist.get_rank() == 0 or dist.get_rank() == dist.get_world_size()-1:
                x = torch.randint(2, self.vocab_size, self.shape)
                y = torch.zeros_like(x)
                y[:-1] = x[1:]
                y[-1] = 0
                return x, y
            else:
                return 0, 0


        if dist.get_world_size() == 1:
            x = torch.randint(2, self.vocab_size, self.shape)
            y = torch.zeros_like(x)
            y[:-1] = x[1:]
            y[-1] = 0
            return x, y
            
        if dist.get_rank() == 0 or dist.get_rank() == dist.get_world_size()-1:
            x = torch.randint(2, self.vocab_size, self.shape)
            if dist.get_rank() == 0:
                return x, 0
            else:
                y = torch.zeros_like(x)
                y[:-1] = x[1:]
                y[-1] = 0
                return 0, y
        else:
            return 0, 0
                
        

class DummyDataFloat(Dataset):
    def __init__(self, length, shape):
        self.length = length
        self.shape = shape
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return torch.rand(size=self.shape)

class DummyImage(Dataset):
    def __init__(self, length, shape, num_classes):
        self.length = length
        self.shape = shape
        self.num_classes = num_classes
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randint(0, 255, self.shape, dtype=torch.float32) / 255.
    
    

def get_train_dataloader(length, shape, gbs, dtype=int, accum_freq=1, shuffle=True, vocab_size = -1,  device="cuda", chimera=False):
    if chimera:
        gbs = gbs // 2
    if dtype == int:
        dataset = DummyDataInt(length, shape, vocab_size, chimera=chimera)
    elif dtype == float:
        dataset = DummyDataFloat(length, shape)
    elif dtype == "image":
        dataset = DummyImage(length, shape, vocab_size)
    
    local_bs = gbs // accum_freq

    return DataLoader(dataset, 
                      batch_size=local_bs,
                      shuffle=shuffle,
                      drop_last=True,
                      pin_memory= True if device == "cuda" else False,
                      )

def get_val_dataloader(length, shape, gbs, dtype=int,  accum_freq=1, shuffle=False, device="cuda"):
   
    local_bs = gbs  // accum_freq

    if dtype == int:
        dataset = DummyDataInt(length, shape)
    elif dtype == float:
        dataset = DummyDataFloat(length, shape)
    else:
        raise NotImplementedError("Only int and float datasets supported")

    return DataLoader(dataset, 
                      batch_size=local_bs,
                      shuffle=shuffle,
                      drop_last=True,
                      pin_memory= True if device == "cuda" else False,
                      )

if __name__ == "__main__":
    train_dataloader = get_train_dataloader(length=128, shape=(3, 224, 224), gbs=32)
    for batch in train_dataloader:
        print(batch.shape)
        

    val_dataloader = get_val_dataloader(length=128, shape=(3, 224, 224), gbs=32)
    for batch in val_dataloader:
        print(batch.shape)
        