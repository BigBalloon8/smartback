import torch.distributed as dist
import torch
from layers import Layer

class BaseSequentialMP:
    def __init__(self, *args):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        num_layers = self.layers // self.world_size
        for i in range(self.world_size):
            if i == self.rank:
                idx = i*self.world_size
                self.layers = args[idx:idx+num_layers]
                if i == self.world_size -1:
                    self.layers.append(args[idx+num_layers:])
        if self.rank != 0:
            self.x = torch.zeros_like(self.layers[0].inputs)
        if self.rank != self.world_size -1:
            self.dL_dout = torch.zeros_like(self.layers[0].out)
        
    def forward(self, x):
        if self.rank == 0:
            for layer in self.layers:
                x = layer(x)
            dist.send(x, self.rank+1)
            dist.barrier()
        elif self.rank == self.world_size-1:
            dist.recv(self.x, self.rank-1)
            x = self.x
            for layer in self.layers:
                x = layer(x)
            dist.barrier()
            return x
        else:
            dist.recv(self.x, self.rank-1)
            x = self.x
            for layer in self.layers:
                x = layer(x)
            dist.send(x, self.rank+1)
            dist.barrier()
    
    def backward(self, dL_dout):
        if self.rank == self.world_size-1:
            for layer in self.layers[::-1]:
                dL_dout = layer.backward(dL_dout)
            dist.send(dL_dout, self.rank-1)
            dist.barrier()
        elif self.rank == 0:
            dist.recv(self.dL_dout, self.rank+1)
            dL_dout = self.dL_dout
            for layer in self.layers[::-1]:
                dL_dout = layer.backward(dL_dout)
            dist.barrier()
        else:
            dist.recv(self.dL_dout, self.rank+1)
            dL_dout = self.dL_dout
            for layer in self.layers[::-1]:
                dL_dout = layer.backward(dL_dout)
            dist.send(dL_dout, self.rank-1)
            dist.barrier()
    
    def update(self):
        for layer in self.layers:
            layer.update(layer)
    
    def to(self):
        ...

class CustomMSequentialMP:
    def __init__(self, *args: Layer):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.layers = []
        self.streams = []
        num_local_layers = len(args) // self.world_size
        for i in range(self.world_size):
            if i == self.rank:
                idx = i*self.world_size
                self.layers = args[idx:idx+num_local_layers]
                if i == self.world_size -1:
                    self.layers.append(args[idx+num_local_layers:])
        if self.rank != 0:
            self.x = torch.zeros_like(self.layers[0].inputs)
        if self.rank != self.world_size -1:
            self.dL_dout = torch.zeros_like(self.layers[0].out)
        for _ in range(len(self.layers)):
            self.streams.append(torch.cuda.Stream())
            
        
    def forward(self, x):
        if self.rank == 0:
            for layer in self.layers:
                x = layer(x)
            dist.send(x, self.rank+1)
            dist.barrier()
        elif self.rank == self.world_size-1:
            dist.recv(self.x, self.rank-1)
            x = self.x
            for layer in self.layers:
                x = layer(x)
            dist.barrier()
            return x
        else:
            dist.recv(self.x, self.rank-1)
            x = self.x
            for layer in self.layers:
                x = layer(x)
            dist.send(x, self.rank+1)
            dist.barrier()
    
    def backward(self, dL_dout):
        if self.rank == self.world_size-1:
            for layer in self.layers[::-1]:
                dL_dout = layer.backward_p1(dL_dout)
            dist.send(dL_dout, self.rank-1)
            for i in range(len(self.layers)):
                with torch.cuda.stream(self.streams[i]):
                    self.layers[i].backward_p2()
            dist.barrier()
        elif self.rank == 0:
            dist.recv(self.dL_dout, self.rank+1)
            dL_dout = self.dL_dout
            for i, layer in enumerate(self.layers[::-1]):
                dL_dout = layer.backward_p1(dL_dout)
                with torch.cuda.stream(self.streams[i]):
                    dL_dout = layer.backward_p2()
            dist.barrier()
        else:
            dist.recv(self.dL_dout, self.rank+1)
            dL_dout = self.dL_dout
            for layer in self.layers[::-1]:
                dL_dout = layer(dL_dout)
            dist.send(dL_dout, self.rank-1)
            for i in range(len(self.layers)):
                with torch.cuda.stream(self.streams[i]):
                    self.layers[i].backward_p2()
            dist.barrier()
    
    def update(self):
        for layer in self.layers:
            layer.update(layer)
    
    def to(self):
        ...