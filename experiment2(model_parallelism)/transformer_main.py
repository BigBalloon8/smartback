import os

import torch
import torch.distributed as dist

from models import Transformer
from data.dummy import get_train_dataloader, get_val_dataloader
import loss
import optimizers

def main():
    torch.manual_seed(31082005)
    torch.cuda.manual_seed(31082005)

    
    dist.init_process_group(backend='gloo')
    if dist.get_world_size() > 4:
        raise NotImplementedError("nccl backend doesnt support multi node")
    if dist.get_world_size() > 1 and dist.get_backend() != "gloo":
        torch.cuda.set_device(dist.get_rank())
    
    gbs = 4
    dim = 2048
    max_seqlen = 1024
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
    
    train_data = get_train_dataloader(65536, (max_seqlen,), gbs, vocab_size=vocab_size)
    
    
    opt = optimizers.SGD(model, 0.0001)
    
    #print(model.layers[1].ff.linears[0])
    #print(dir(model.layers[1].ff.linears[0]))
    
    for data in train_data:
        if dist.get_rank() != 0:
            data = None
        y = model.forward(data)
        break
    if dist.get_rank() == dist.get_world_size()-1:
        dL_dout = torch.ones_like(y)
    else:
        dL_dout = None
        
        
    model.backward(dL_dout)
    
    model.update()
    


    


if __name__ == "__main__":
    main()