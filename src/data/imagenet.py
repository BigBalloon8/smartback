import os
import random

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import torchvision


def get_train_dataloader(data_path, gbs, accum_freq=1, subset=None, shuffle=True, prefetch=4, device="cuda"):

    path = os.path.join(data_path, "train")

    local_bs = (gbs // dist.get_world_size()) // accum_freq

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    dataset = ImageFolder(root=path,
                          transform=transform)

    if subset:
        indices = random.sample(range(len(dataset)), subset)
        dataset = Subset(dataset, indices)

    if dist.get_world_size() > 1:
        sampler = DistributedSampler(dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    else:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

    return DataLoader(dataset, 
                      sampler=sampler,
                      batch_size=local_bs, 
                      drop_last=True,
                      num_workers=4,
                      prefetch_factor=prefetch,
                      pin_memory = True if device == "cuda" else False 
                      )

def get_val_dataloader(data_path, gbs, accum_freq=1, subset=None, shuffle=False, prefetch=4, device="cuda"):
    
    path = os.path.join(data_path, "val")

    local_bs = (gbs // dist.get_world_size()) // accum_freq

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                         ])
    dataset = ImageFolder(root=path,
                          transform=transform)
    if subset:
        indices = random.sample(range(len(dataset)), subset)
        dataset = Subset(dataset, indices)
    if dist.get_world_size() > 1:
        sampler = DistributedSampler(dataset, dist.get_world_size(), dist.get_rank())
    else:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

    return DataLoader(dataset, 
                      sampler=sampler,
                      batch_size=local_bs, 
                      drop_last=True,
                      num_workers=1,
                      prefetch_factor=prefetch,
                      pin_memory = True if device == "cuda" else False 
                      )

