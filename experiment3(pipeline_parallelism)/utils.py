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

def calc_times(t0,t1,t2,bp2_times):
    group = dist.new_group([0, dist.get_world_size()-1])
    if dist.get_rank() == dist.get_world_size()-1:
        t1_cp = t1.clone()
        dist.reduce(t1, dist.get_world_size()-1, group=group)
        dist.reduce(-t1_cp, 0, group=group)
        print(f"Avg Fwd step time: {torch.mean(t1.to(torch.float64))*1e-9}") 
    elif dist.get_rank() == 0:
        dist.reduce(-t0, dist.get_world_size()-1, group=group)
        dist.reduce(t2, 0, group=group)
        print(f"Avg Bwd step time: {torch.mean(t2.to(torch.float64))*1e-9}")
    dist.reduce(bp2_times, 0)
    if dist.get_rank() == 0:
        print(f"Avg Bwd_p2 step time: {torch.mean(bp2_times.to(torch.float64))*1e-9}")
    