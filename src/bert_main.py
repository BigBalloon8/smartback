import os
import time
import click

import torch
import torch.distributed as dist
from tqdm import tqdm

from models import BertLarge
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
    vocab_size = 30522
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    model = BertLarge(pipe_algo=algo, device=device, dtype=dtype)
    
    model.init_params(gbs, (512, 1024))
    model.multi_stage(msbp)
    
    train_data = get_train_dataloader(512, (512,), gbs, vocab_size=vocab_size, dtype="bert")
    
    n_params = model.get_num_params()
    
    opt = optimizers.Adam(model, 0.0001)

    param_mem = share_var(torch.cuda.memory_allocated())

    if dist.get_rank() == 0:
        print(f"Model: {model.__class__.__name__}")
        print(f"No. Params: {n_params:,}")
        print(f"Pipe Algo: {model.pipe_algo}")
        print(f"Global Batch Size: {gbs}")
        print(f"Multi Stage: {model.layers[0].multi_stage}")
        print(f"Memory Parameters: {param_mem/1024**3:.4f}GB")
        print(f"Dtype: {model.dtype}")

    for x, y_ns, y_mask, m, sm in train_data:
        x, m, sm = x.to(device), m.to(device), sm.to(device)
        y_ns, y_mask = y_ns.to(device), y_mask.to(device)
        losses = model.train_step((x, sm), (y_ns, y_mask, m))
        model.update()
        model.zero_grad()
        break
    
    torch.cuda.reset_peak_memory_stats()
    with benchmark("Time For Epoch", 512*512):
        for x, y_ns, y_mask, m, sm in train_data:
            x, m, sm = x.to(device), m.to(device), sm.to(device)
            y_ns, y_mask = y_ns.to(device), y_mask.to(device)
            losses = model.train_step((x, sm), (y_ns, y_mask, m))
            model.update()
            model.zero_grad()

    max_mem = share_var(torch.cuda.max_memory_allocated(), op=dist.ReduceOp.MAX)
    #calc_times(torch.tensor(model.t0,device="cuda"), torch.tensor(model.t1,device="cuda"), torch.tensor(model.t2,device="cuda"), torch.tensor(model.bwd_p2_times,device="cuda"))

    if dist.get_rank() == 0:
        print(f"Memory Required: {max_mem/1024**3:.4f}GB")
        print(f"---------------------------------------------")
    
    time.sleep(1)
    exit()

if __name__ == "__main__":
    with torch.inference_mode():
        main()