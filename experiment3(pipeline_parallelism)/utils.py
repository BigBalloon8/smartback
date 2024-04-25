import time
import contextlib

import torch
import torch.distributed as dist

@contextlib.contextmanager
def benchmark(name=None, dset_size=None):
    if name is None:
        name = "Elapsed Time"
    start_time = time.time()
    yield
    end_time = time.time()
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"{name}: {end_time - start_time :.2f}s")
            if dset_size is not None:
                print(f"Throughput: {dset_size / (end_time - start_time):.2f} samples/s")
    else:
        print(f"{name}: {end_time - start_time :.2f}s")
        if dset_size is not None:
            print(f"Throughput: {dset_size / (end_time - start_time):.2f} samples/s")

def share_var(x:float, mean=False, op=dist.ReduceOp.SUM):
    if dist.get_backend() == "nccl":
        device = "cuda"
    else:
        device = "cpu"
    tensor = torch.tensor(x, device=device).clone()
    dist.all_reduce(tensor, op=op)
    if mean:
        tensor /= dist.get_world_size()
    return tensor.item()
    