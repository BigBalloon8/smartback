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
    print(torch.__version__)
    torch.manual_seed(31082005)
    torch.cuda.manual_seed(31082005)
    #os.environ["RANK"] = "0"
    #os.environ["WORLD_SIZE"] = "1"
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend='nccl')
    if dist.get_world_size() > 4:
        raise NotImplementedError("nccl backend doesnt support multi node")
    if dist.get_world_size() > 1 and dist.get_backend() != "gloo":
        pass
    torch.cuda.set_device(dist.get_rank())
    
    gbs = 16
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
        pipe_algo="1f1b",
        device=device
    )
    
    model.init_params(gbs, (max_seqlen, dim))
    model.multi_stage(False)

    train_data = get_train_dataloader(gbs*128, (max_seqlen,), gbs, vocab_size=vocab_size)

    n_params = model.get_num_params()

    if dist.get_rank() == 0:
        print(f"No. Params: {n_params:,}")
        print(model.layers[0].multi_stage)
    
    opt = optimizers.Adam(model, 0.0001)

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
    print(dir(torch.cuda.memory))
    #torch.cuda.memory._dump_snapshot("/mnt/ceph_rbd/transformer_out.pickle")
    

if __name__ == "__main__":
    with torch.inference_mode():
        main()