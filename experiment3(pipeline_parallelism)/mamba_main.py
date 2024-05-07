import os
import time
import click

import torch
import torch.distributed as dist
from tqdm import tqdm

from models import Mamba
from data.dummy import get_train_dataloader, get_val_dataloader
import loss
import optimizers
from utils import benchmark, share_var, calc_times

@click.command()
@click.option("--msbp", is_flag=True, default=False)
@click.option("--algo", default="none")
def main(msbp, algo):
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
    
    gbs = 16 if algo == "1f1b-2" else 8
    dim = 2048
    max_seqlen = 1024
    vocab_size = 32000
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = Mamba(
        num_blocks=48,
        d_model=dim,
        vocab_size=vocab_size,
        pipe_algo=algo,
        criterion=loss.NLPCrossEntropyLoss(),
        device=device,
        dtype=torch.float16
    )
    
    model.init_params(gbs, (max_seqlen, dim))
    model.multi_stage(msbp)

    train_data = get_train_dataloader(128, (max_seqlen,), gbs, vocab_size=vocab_size)
    
    n_params = model.get_num_params()
    
    opt = optimizers.AdamW(model, 0.0001)

    param_mem = share_var(torch.cuda.memory_allocated())

    if dist.get_rank() == 0:
        print(f"Model: {model.__class__.__name__}")
        print(f"No. Params: {n_params:,}")
        print(f"Pipe Algo: {model.pipe_algo}")
        print(f"Global Batch Size: {gbs}")
        print(f"Multi Stage: {model.layers[0].multi_stage}")
        print(f"Memory Parameters: {param_mem/1024**3:.4f}GB")
        print(f"Dtype: {model.layers[0].weights.dtype}")

    for x,y in train_data:
        x, y = x.to(device), y.to(device)
        losses = model.train_step(x, y)
        model.update()
        model.zero_grad()
        break

    torch.cuda.reset_peak_memory_stats()
    with benchmark("Time For Epoch", 128*max_seqlen):
        for x,y in train_data:
            x, y = x.to(device), y.to(device)
            losses = model.train_step(x, y)
            model.update()
            model.zero_grad()
    
    max_mem = share_var(torch.cuda.max_memory_reserved(), op=dist.ReduceOp.MAX)
    #calc_times(torch.tensor(model.t0,device="cuda"), torch.tensor(model.t1,device="cuda"), torch.tensor(model.t2,device="cuda"), torch.tensor(model.bwd_p2_times,device="cuda"))

    if dist.get_rank() == 0:
        print(f"Memory Required: {max_mem/1024**3:.4f}GB")
        print(f"---------------------------------------------")
    
    
    torch.cuda.cudart().cudaProfilerStop()
    time.sleep(1)
    exit()

if __name__ == "__main__":
    with torch.inference_mode():
        main()