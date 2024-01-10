from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

import layers

class Model(ABC):
    def __init__(self):
        self.layers: list[layers.Layer] = []
        self.streams: list[torch.cuda.Stream] = []
        self.x = ...
        self.dL_dout = ...

    def forward(self, input):
        if dist.get_rank()  == 0:
            for layer in self.layers:
                input = layer(input)
            dist.send(input, dist.get_rank()+1)
            dist.barrier()
        elif dist.get_rank() == dist.get_world_size()-1:
            dist.recv(..., dist.get_rank()-1)
            x = self.x
            for layer in self.layers:
                x = layer(x)
            dist.barrier()
            return x
        else:
            dist.recv(self.x, dist.get_rank()-1)
            x = self.x
            for layer in self.layers:
                x = layer(x)
            dist.send(x, dist.get_rank()+1)
            dist.barrier()

    def backward(self, dL_dout):
        if dist.get_rank() == dist.get_world_size()-1:
            for layer in self.layers[::-1]:
                dL_dout = layer.backward_p1(dL_dout)
            dist.send(dL_dout, dist.get_rank()-1)
            for i in range(len(self.layers)):
                with torch.cuda.stream(self.streams[i]):
                    self.layers[i].backward_p2()
            dist.barrier()
        elif dist.get_rank() == 0:
            dist.recv(self.dL_dout, dist.get_rank()+1)
            dL_dout = self.dL_dout
            for i, layer in enumerate(self.layers[::-1]):
                dL_dout = layer.backward_p1(dL_dout)
                with torch.cuda.stream(self.streams[i]):
                    dL_dout = layer.backward_p2()
            dist.barrier()
        else:
            dist.recv(self.dL_dout, dist.get_rank()+1)
            dL_dout = self.dL_dout
            for layer in self.layers[::-1]:
                dL_dout = layer(dL_dout)
            dist.send(dL_dout, self.rank-1)
            for i in range(len(self.layers)):
                with torch.cuda.stream(self.streams[i]):
                    self.layers[i].backward_p2()
            dist.barrier()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    
class Transformer(Model):
    def __init__(self, dim_size, num_heads, num_kv_heads, max_seqlen, num_blocks, device):
        self.block_kwargs = {
            "dim_size": dim_size,
            "num_heads": num_heads, 
            "num_kv_heads": num_kv_heads, 
            "max_seqlen": max_seqlen, 
            "device": device
            }
        self.num_blocks = num_blocks
    
    def init_params(self, x):
        self.layers = []
        num_local_blocks = self.num_blocks // dist.get_world_size()
        for _ in range(num_local_blocks):
            layer = layers.TransformerPPBlock(**self.block_kwargs)
            layer.init_params()
            self.layers.append(layer)
        self.streams = [torch.cuda.Stream() for _ in range(num_local_blocks)]
        if dist.get_rank() == 0:
            self.layers.insert(0, ...)  # TODO implement embeddings
        
        
