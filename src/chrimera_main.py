import os
import time

import torch
import torch.distributed as dist

from chimera import Transformer
from data.dummy import get_train_dataloader, get_val_dataloader
import layers
import loss
import optimizers
from utils import benchmark

import warnings
warnings.filterwarnings("ignore")


def main():
    torch.manual_seed(31082005)
    torch.cuda.manual_seed(31082005)
    #os.environ["RANK"] = "0"
    #os.environ["WORLD_SIZE"] = "1"
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend='nccl')
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
        num_heads=16,
        num_kv_heads=8,
        num_blocks=16,
        max_seqlen=max_seqlen,
        vocab_size=vocab_size,
        criterion=loss.NLPCrossEntropyLoss(),
        device=device
    )
    
    model.init_params(gbs, (max_seqlen, dim))
    model.multi_stage(True)

    train_data = get_train_dataloader(gbs*32, (max_seqlen,), gbs, vocab_size=vocab_size, chimera=True)


    if dist.get_rank() == 0:
        print(model._2bp)
    
    opt = optimizers.SGD(model, 0.0001, chimera=True)

    torch.cuda.cudart().cudaProfilerStart()
    
    with benchmark("Time For Epoch"):
        for x,y in train_data:
            x ,y = x.to(device), y.to(device)
            with benchmark("Time For Step"):
                losses = model.train_step(x, y)
                #if dist.get_rank() == dist.get_world_size()-1:
                    #print(f"Loss: {losses}")
                model.update()
                model.zero_grad()
    
    
    torch.cuda.cudart().cudaProfilerStop()
    time.sleep(1)
    exit()
    #torch.cuda.memory._dump_snapshot("/mnt/ceph_rbd/transformer_out.pickle")
    

if __name__ == "__main__":
    with torch.inference_mode():
        main()