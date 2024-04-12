import os
import time
import click

import torch
import torch.distributed as dist
from tqdm import tqdm

from models import Transformer
from data.dummy import get_train_dataloader, get_val_dataloader
import layers
import loss
import optimizers
from utils import benchmark

@click.command()
@click.option("--msbp", is_flag=True, default=False)
def main(msbp):
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
    dim = 4096
    max_seqlen = 1024
    vocab_size = 32000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #torch.cuda.memory._record_memory_history(True)

    model = Transformer(
        dim_size=dim,
        num_heads=32,
        num_kv_heads=32,
        num_blocks=32,
        max_seqlen=max_seqlen,
        vocab_size=vocab_size,
        criterion=loss.NLPCrossEntropyLoss(),
        pipe_algo="gpipe",
        device=device,
        dtype=torch.float16,
    )
    
    model.init_params(gbs, (max_seqlen, dim))
    model.multi_stage(msbp)

    train_data = get_train_dataloader(gbs*32, (max_seqlen,), gbs, vocab_size=vocab_size)
    if dist.get_rank() == 0:
        train_data = tqdm(train_data, unit_scale=gbs*max_seqlen)

    n_params = model.get_num_params()
    
    opt = optimizers.Adam(model, 0.0001)

    if dist.get_rank() == 0:
        print(f"Model: {model.__class__.__name__}")
        print(f"No. Params: {n_params:,}")
        print(f"Pipe Algo: {model.pipe_algo}")
        print(f"Multi Stage: {model.layers[0].multi_stage}")
        print(f"Memory Parameters: {torch.cuda.memory_allocated()/1e9:.4f}GB")
        print(f"dtype: {model.layers[0].weights.dtype}")

    
    torch.cuda.cudart().cudaProfilerStart()

    # Warmup for compilation
    for x, y in train_data:
        x ,y = x.to(device), y.to(device)
        model.train_step(x, y)
        model.update()
        model.zero_grad()
        break
    
    model.zero_grad()
    model.zero_act()

    
    with benchmark("Time For Epoch"):
        for x,y in train_data:
            #with benchmark("Time For Batch"):
            x ,y = x.to(device), y.to(device)
            losses = model.train_step(x, y)
            model.update()
            model.zero_grad()
    
    if dist.get_rank() == 0:
        print(f"Memory Required: {torch.cuda.max_memory_allocated()/1e9:.4f}GB")
    
    
    torch.cuda.cudart().cudaProfilerStop()
    exit()
    print(dir(torch.cuda.memory))
    #torch.cuda.memory._dump_snapshot("/mnt/ceph_rbd/transformer_out.pickle")
    

if __name__ == "__main__":
    with torch.inference_mode():
        main()