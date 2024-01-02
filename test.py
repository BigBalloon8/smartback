from mpi4py import MPI 
import torch
import torch.distributed as dist

dist.init_process_group(backend='mpi')

comm = MPI.COMM_WORLD

print(comm.Get_size())

