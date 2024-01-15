import os
import time

import torch
import torch.distributed as dist

from models import ResNet50
from data.dummy import get_train_dataloader, get_val_dataloader
import layers
import loss
import optimizers
from utils import benchmark

def main():
    torch.manual_seed(31082005)
    torch.cuda.manual_seed(31082005)
    
    dist.init_process_group(backend='gloo')
    if dist.get_world_size() > 4:
        raise NotImplementedError("nccl backend doesnt support multi node")
    if dist.get_world_size() > 1 and dist.get_backend() != "gloo":
        torch.cuda.set_device(dist.get_rank())
    
    gbs = 16
    num_classes = 1000
    device = "cpu"
    
    model = ResNet50(num_classes=num_classes, device=device)
    model.init_params(gbs)
    model.multi_stage(True)
    
    train_data = get_train_dataloader(65536, (3,244,244), gbs, dtype="image")
    
    opt = optimizers.SGD(model, 0.0001)
    
    for data in train_data:
        data = data.to(device)
        with benchmark("Time For Step"):
            if dist.get_rank() != 0:
                data = None
            y = model.forward(data)
            print(f"Rank {dist.get_rank()}: Forward Complete")
            if dist.get_rank() == dist.get_world_size()-1:
                dL_dout = torch.ones_like(y)
            else:
                dL_dout = None
            model.backward(dL_dout)
            print(f"Rank {dist.get_rank()}: Backward Complete")
        model.update()
        model.zero_grad()
        
if __name__ == "__main__":
    with torch.inference_mode():
        main()