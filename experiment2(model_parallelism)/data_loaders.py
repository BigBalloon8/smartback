import os
import torch

def empty_data_loader(num_iter):
    idx = 0
    while idx < num_iter:
        yield None, None
        idx += 1

class MPCustomDataLoader:
    def __init__(self, dataset, batch_size) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.data = dataset
        self.batch_size = batch_size
    
    def __next__(self):
        if self.local_rank == 0:
            for i in range(len(self.data)//self.batch_size):
                start_idx = i*self.batch_size
                yield zip(*[self.data[start_idx+b] for b in range(self.batch_size)])
                
    
    def __iter__(self):
        return self