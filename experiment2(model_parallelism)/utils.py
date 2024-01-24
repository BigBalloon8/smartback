import time
import contextlib

import torch.distributed as dist

@contextlib.contextmanager
def benchmark(name=None):
    if name is None:
        name = "Elapsed Time"
    start_time = time.time()
    yield
    end_time = time.time()
    if dist.get_rank() == 0:
        print(f"{name}: {end_time - start_time :.2f}s")
    