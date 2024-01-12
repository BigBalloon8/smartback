import os

import torch
import torch.distributed as dist

from models import Transformer
import loss
import optimizers

def main():
    torch.manual_seed(31082005)
    torch.cuda.manual_seed(31082005)

    dist.init_process_group(backend='mpi')
    if dist.get_world_size() > 4:
        raise NotImplementedError("nccl backend doesnt support multi node")
    if dist.get_world_size() > 1:
        torch.cuda.set_device(dist.get_rank())
    
    model = Transformer(
        dim_size=2048,
        num_heads=16,
        num_kv_heads=8,
        num_blocks=16,
        max_seqlen=1024
    )


    


if __name__ == "__main__":
    main()