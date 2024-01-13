import torch
import time

a = torch.randn(10000)
b = torch.randn(10000,10000)

t = time.time_ns()
torch.matmul(a, b)
print((time.time_ns() - t)*1e-09)

t = time.time_ns()
torch.mm(a.unsqueeze(0), b).squeeze()
print((time.time_ns() - t)*1e-09)

t = time.time_ns()
a @ b
print((time.time_ns() - t)*1e-09)

t = time.time_ns()
torch.einsum("i,ij->j",a,b)
print((time.time_ns() - t)*1e-09)
