import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from models import Mamba
from data.dummy import get_train_dataloader, get_val_dataloader
import loss
import optimizers
from utils import benchmark

def main():
    torch.manual_seed(31082005)
    torch.cuda.manual_seed(31082005)
    #os.environ["RANK"] = "0"
    #os.environ["WORLD_SIZE"] = "1"
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend='nccl')
    if dist.get_world_size() > 8:
        raise NotImplementedError("nccl backend doesnt support multi node")
    if dist.get_world_size() > 1 and dist.get_backend() != "gloo":
        pass
    torch.cuda.set_device(dist.get_rank())
    
    gbs = 8
    dim = 2048
    max_seqlen = 1024
    vocab_size = 32000
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = Mamba(
        num_blocks=48,
        d_model=dim,
        vocab_size=vocab_size,
        pipe_algo="1f1b-1",
        criterion=loss.NLPCrossEntropyLoss(),
        device=device,
        dtype=torch.bfloat16
    )
    
    model.init_params(gbs, (max_seqlen, dim))
    model.multi_stage(True)

    train_data = get_train_dataloader(gbs*16, (max_seqlen,), gbs, vocab_size=vocab_size)
    if dist.get_rank() == 0:
        train_data = tqdm(train_data, unit_scale=gbs*max_seqlen)
    
    n_params = model.get_num_params()
    
    opt = optimizers.AdamW(model, 0.0001)

    if dist.get_rank() == 0:
        print(f"Model: {model.__class__.__name__}")
        print(f"No. Params: {n_params:,}")
        print(f"Pipe Algo: {model.pipe_algo}")
        print(f"Multi Stage: {model.layers[0].multi_stage}")
        print(f"Memory Parameters: {torch.cuda.memory_allocated()/1e9:.4f}GB")
        print(f"Dtype: {model.layers[0].weights.dtype}")

    torch.cuda.cudart().cudaProfilerStart()

    for x,y in train_data:
        x, y = x.to(device), y.to(device)
        losses = model.train_step(x, y)
        model.update()
        model.zero_grad()
        break
    
    with benchmark("Time For Epoch"):
        for x,y in train_data:
            x, y = x.to(device), y.to(device)
            losses = model.train_step(x, y)
            model.update()
            model.zero_grad()
    
    if dist.get_rank() == 0:
        print(f"Memory Required: {torch.cuda.max_memory_allocated()/1e9:.4f}GB")
    
    
    torch.cuda.cudart().cudaProfilerStop()
    time.sleep(1)
    exit()

if __name__ == "__main__":
    with torch.inference_mode():
        main()