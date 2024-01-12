import random

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class DummyDataInt(Dataset):
    def __init__(self, length, shape):
        self.length = length
        self.shape = shape
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return torch.randint(size=self.shape)

class DummyDataFloat(Dataset):
    def __init__(self, length, shape):
        self.length = length
        self.shape = shape
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return torch.rand(size=self.shape)

def get_train_dataloader(length, shape, gbs, accum_freq=1, shuffle=True, device="cuda"):
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    local_bs = (gbs // world_size) // accum_freq

    dataset = DummyDataInt(length, shape)

    if world_size > 1:
        sampler = DistributedSampler(dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    else:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

    return DataLoader(DummyDataFloat(length, shape), 
                      batch_size=local_bs,
                      sampler=sampler,
                      drop_last=True,
                      pin_memory= True if device == "cuda" else False,
                      )

def get_val_dataloader(length, shape, gbs, type=int,  accum_freq=1, shuffle=False, device="cuda"):
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    local_bs = (gbs // world_size) // accum_freq

    if type == int:
        dataset = DummyDataInt(length, shape)
    elif type == float:
        dataset = DummyDataFloat(length, shape)
    else:
        raise NotImplementedError("Only int and float datasets supported")
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    else:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

    return DataLoader(DummyDataFloat(length, shape), 
                      batch_size=local_bs,
                      sampler=sampler,
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
        