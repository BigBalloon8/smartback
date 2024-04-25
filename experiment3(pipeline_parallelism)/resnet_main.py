import os
import time
import click

import torch
import torch.distributed as dist

torch.backends.cudnn.benchmark = True

from models import ResNet152
from data.dummy import get_train_dataloader, get_val_dataloader
import layers
import loss
import optimizers
from utils import benchmark, share_var

@click.command()
@click.option("--msbp", is_flag=True, default=False)
@click.option("--algo", default="none")
def main(msbp, algo):
    torch.manual_seed(31082005)
    torch.cuda.manual_seed(31082005)
    
    dist.init_process_group(backend='nccl')
    if dist.get_world_size() > 8:
        raise NotImplementedError("nccl backend doesnt support multi node")
    if dist.get_world_size() > 1 and dist.get_backend() != "gloo":
        pass
    torch.cuda.set_device(dist.get_rank())
    
    gbs = 32
    num_classes = 1000
    device = "cuda"
    dtype = torch.float32
    
    model = ResNet152(num_classes=num_classes, device=device, pipe_algo=algo, criterion=loss.CrossEntropyLoss(), dtype=dtype)
    model.init_params(gbs)
    model.multi_stage(msbp)
    
    train_data = get_train_dataloader(2048, (3,224,224), gbs, vocab_size=1000, dtype="image")
    
    n_params = model.get_num_params()

    opt = optimizers.SGD(model, 0.0001)

    if dist.get_rank() == 0:
        print(f"Model: {model.__class__.__name__}")
        print(f"No. Params: {n_params:,}")
        print(f"Pipe Algo: {model.pipe_algo}")
        print(f"Global Batch Size: {gbs}")
        print(f"Multi Stage: {model.layers[0].multi_stage}")
        print(f"Memory Parameters: {torch.cuda.memory_allocated()/1024**3:.4f}GB")
        print(f"Dtype: {model.dtype}")

    for x, y in train_data:
        x, y = x.to(dtype).to(device), y.to(dtype).to(device)
        losses = model.train_step(x, y)
        model.update()
        model.zero_grad()
        break
    
    with benchmark("Time For Epoch", 2048):
        for x, y in train_data:
            x, y = x.to(dtype).to(device), y.to(dtype).to(device)
            losses = model.train_step(x, y)
            model.update()
            model.zero_grad()
            if dist.get_rank() == 0:
                print(f"Memory Required: {torch.cuda.max_memory_allocated()/1024**3:.4f}GB")
    
    max_mem = share_var(torch.cuda.max_memory_allocated(), op=dist.ReduceOp.MAX)
    if dist.get_rank() == 1:
        print(f"Memory Required: {max_mem/1024**3:.4f}GB")

        
if __name__ == "__main__":
    with torch.no_grad():
        main()