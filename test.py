import torch
import time

def f(x,z, n):
    n2 = n**2
    return n2*x*z - 1/n

def g(x,z,n,n2):
    return n2*x*z - 1/n

s = 1000000
x, z, n, n2 = torch.randn(s), torch.randn(s), torch.tensor(s), torch.tensor(s)**2

start = time.time_ns()
torch.vmap(f, in_dims=(0,0,None))(x,z,n)
int = time.time_ns()
torch.vmap(g, in_dims=(0,0,None, None))(x, z, n, n2)
end = time.time_ns()

print((int - start)*10**-9)
print((end - int)*10**-9)
