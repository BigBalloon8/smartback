import os
import time

import torch
import torch.distributed as dist

from models import Transformer
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
    
    gbs = 1
    dim = 2038*2
    max_seqlen = 512
    vocab_size = 32000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Transformer(
        dim_size=dim,
        num_heads=16,
        num_kv_heads=8,
        num_blocks=12,
        max_seqlen=max_seqlen,
        vocab_size=vocab_size,
        device=device
    )
    
    model.init_params((gbs, max_seqlen, dim))
    model.multi_stage(True)
    
    train_data = get_train_dataloader(16, (max_seqlen,), gbs, vocab_size=vocab_size)

    print(model.layers[0].multi_stage)
    
    opt = optimizers.Adam(model, 0.0001)

    torch.cuda.cudart().cudaProfilerStart()
    
    with benchmark("Time For Epoch"):
        for data in train_data:
            data = data.to(device)
            with benchmark("Time For Step"):
                if dist.get_rank() != 0:
                    data = None
                y = model.forward(data)
                if dist.get_rank() == dist.get_world_size()-1:
                    dL_dout = torch.ones_like(y)
                else:
                    dL_dout = None
                model.backward(dL_dout)
                #torch.cuda.synchronize()
                model.update()
                #torch.cuda.synchronize()
                model.zero_grad()
                #dist.barrier()
            

    torch.cuda.cudart().cudaProfilerStop()
    

if __name__ == "__main__":
    with torch.inference_mode():
        main()