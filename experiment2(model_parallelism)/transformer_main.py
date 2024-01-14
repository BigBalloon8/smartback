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

    
    dist.init_process_group(backend='gloo')
    if dist.get_world_size() > 4:
        raise NotImplementedError("nccl backend doesnt support multi node")
    if dist.get_world_size() > 1 and dist.get_backend() != "gloo":
        torch.cuda.set_device(dist.get_rank())
    
    gbs = 16
    dim = 512
    max_seqlen = 256
    vocab_size = 32000
    
    model = Transformer(
        dim_size=dim,
        num_heads=16,
        num_kv_heads=8,
        num_blocks=4,
        max_seqlen=max_seqlen,
        vocab_size=vocab_size,
        device="cpu"
    )
    
    model.init_params((gbs, max_seqlen, dim))
    model.multi_stage(False)
    
    train_data = get_train_dataloader(65536, (max_seqlen,), gbs, vocab_size=vocab_size)
    
    print(model.layers[0].multi_stage)
    
    opt = optimizers.Adam(model, 0.0001)
    
    for data in train_data:
        with benchmark("Time For Step"):
            if dist.get_rank() != 0:
                data = None
            y = model.forward(data)
            if dist.get_rank() == dist.get_world_size()-1:
                dL_dout = torch.ones_like(y)
            else:
                dL_dout = None
            model.backward(dL_dout)
            model.update()
            model.zero_grad()
    
    
    


    


if __name__ == "__main__":
    with torch.inference_mode():
        main()