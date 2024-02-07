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
    
    dist.init_process_group(backend='nccl')
    if dist.get_world_size() > 4:
        raise NotImplementedError("nccl backend doesnt support multi node")
    if dist.get_world_size() > 1 and dist.get_backend() != "gloo":
        pass
    torch.cuda.set_device(dist.get_rank())
    
    gbs = 16
    num_classes = 1000
    device = "cuda"
    
    model = ResNet50(num_classes=num_classes, device=device)
    model.init_params(gbs//dist.get_world_size())
    model.multi_stage(True)
    
    train_data = get_train_dataloader(16*16, (3,224,224), gbs, dtype="image")
    
    opt = optimizers.SGD(model, 0.0001)
    
    with benchmark("Time For Epoch"):
        for data in train_data:
            data = data.to(device)
            with benchmark("Time For Step"):
                if dist.get_rank() != 0:
                    data = None
                y = model.forward(data)
                if dist.get_rank() == dist.get_world_size()-1:
                    dL_dout = torch.ones_like(torch.stack(y))
                else:
                    dL_dout = range(dist.get_world_size())
                model.backward(dL_dout)
                model.update()
                model.zero_grad()
        
if __name__ == "__main__":
    with torch.inference_mode():
        main()