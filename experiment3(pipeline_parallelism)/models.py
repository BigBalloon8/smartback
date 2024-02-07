from typing import Sequence

import torch
import torch.distributed as dist

import layers

class Model:
    def __init__(self):
        self.layers: list[layers.Layer] = []
        self.x = ...
        self.dL_dout = ...
        self.f_recv_buffer = ...
        self.b_recv_buffer = ...
        self.sub_layers: list[layers.Layer] = []
        self.streams: list[torch.cuda.Stream] = []

    def forward(self, input:torch.Tensor):
        if dist.get_rank() == dist.get_world_size()-1:
            outputs = []
        if dist.get_rank() ==0:
            mini_batches = torch.chunk(input, dist.get_world_size())
        else:
            mini_batches = range(dist.get_world_size())
        
        for i, mini_batch in enumerate(mini_batches):
            if dist.get_rank()  == 0:
                torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Forward:{i}")
                x = mini_batch
                for layer in self.layers:
                    x = layer(x)
                dist.send(x, dist.get_rank()+1)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                
            elif dist.get_rank() == dist.get_world_size()-1:
                dist.recv(self.f_recv_buffer, dist.get_rank()-1)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Forward:{i}")
                x = self.f_recv_buffer.clone()
                for layer in self.layers:
                    x = layer(x)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                outputs.append(x)
            
            else:
                dist.recv(self.f_recv_buffer, dist.get_rank()-1)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Forward:{i}")
                x = self.f_recv_buffer.clone()
                for layer in self.layers:
                    x = layer(x)
                dist.send(x, dist.get_rank()+1)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()

        if dist.get_rank() == dist.get_world_size()-1:
            return outputs

    def backward(self, dL_dout_mb: Sequence[torch.Tensor]):
        for i, dL_dout in enumerate(dL_dout_mb):
            if dist.get_rank() == dist.get_world_size()-1:
                torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Backward P1:{i}")
                for layer in self.layers[::-1]:
                    dL_dout = layer.backward_p1(dL_dout)
                torch.cuda.synchronize()
                work = dist.isend(dL_dout, dist.get_rank()-1)
                torch.cuda.nvtx.range_pop()
            
            elif dist.get_rank() == 0:
                dist.recv(self.b_recv_buffer, dist.get_rank()+1)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Backward P1:{i}")
                dL_dout = self.b_recv_buffer.clone()
                for layer in self.layers[::-1]:
                    dL_dout = layer.backward_p1(dL_dout)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            
            else:
                dist.recv(self.b_recv_buffer, dist.get_rank()+1)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Backward P1;{i}")
                dL_dout = self.b_recv_buffer.clone()
                for layer in self.layers[::-1]:
                    dL_dout = layer.backward_p1(dL_dout)
                torch.cuda.synchronize()
                dist.isend(dL_dout, dist.get_rank()-1)
                torch.cuda.nvtx.range_pop()
        
        if self.layers[0].multi_stage:
            torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Backward P2")
            for layer, s in zip(self.sub_layers, self.streams):
                with torch.cuda.stream(s):
                    layer.backward_p2()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        dist.barrier()

    def update(self):
        torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Update Grads")
        for layer, s in zip(self.sub_layers, self.streams):
            with torch.cuda.stream(s):
                layer.update()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.nvtx.range_pop()
    
    def multi_stage(self, _bool):
        for layer in self.layers:
            layer.multi_stage_set(_bool)
    
    def set_batch_size(self, bs):
        for layer in self.layers:
            layer.batch_size_set(bs)

    def zero_grad(self):
        torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Zero Grad")
        for layer in self.layers:
            layer.zero_grad()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.nvtx.range_pop()
    
    def zero_act(self):
        for layer in self.layers:
            layer.clear_acts()

    def save(self, path):
        pass

    def load(self, path):
        pass

    def to(self, arg):
        for layer in self.layers:
            layer.to(arg)
    
    def get_num_params(self):
        tensor_sizes = []
        for layer in self.layers:
            layer.get_num_params(tensor_sizes)
        tensor = torch.tensor(sum(tensor_sizes), device=self.device)
        dist.all_reduce(tensor)
        if dist.get_rank() == 0:
            print(tensor.item())

    
class Transformer(Model):
    def __init__(self, dim_size, num_heads, num_kv_heads, max_seqlen, num_blocks, vocab_size, device):
        self.block_kwargs = {
            "dim": dim_size,
            "num_heads": num_heads, 
            "num_kv_heads": num_kv_heads, 
            "max_seqlen": max_seqlen, 
            "device": device
            }
        self.embedding_kwargs ={
            "num_embeddings": vocab_size,
            "dim": dim_size, 
            "device": device
        }
        self.num_blocks = num_blocks
        self.device = device
        
    
    def init_params(self, mb_input_shape):
        self.layers = []
        num_local_blocks = self.num_blocks // dist.get_world_size()
        for _ in range(num_local_blocks):
            layer = layers.TransformerPPBlock(**self.block_kwargs)
            layer.init_params()
            self.layers.append(layer)
        if dist.get_rank() == 0:
            self.layers.insert(0, layers.Embeddings(**self.embedding_kwargs))
            self.layers[0].init_params()
        
        self.f_recv_buffer = torch.zeros(mb_input_shape, device=self.device)
        self.b_recv_buffer = self.f_recv_buffer

        self.sub_layers = []
        for layer in self.layers:
            layer._get_model_sub_layers(self.sub_layers)   
        
        self.streams = [torch.cuda.Stream() for _ in self.sub_layers]
        torch.cuda.synchronize()
        

class ResNet50(Model):
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
    
    def init_params(self, gmbs):
        model = layers.ResNet(layers.ResNetBottleneck, [3,4,6,3], device=self.device)
        self.layers = []
        self.layers.append(layers.Conv2D(3, 64, 3, stride=1, padding=1, bias=False, device=self.device))
        self.layers.append(layers.BatchNorm2D(64, device=self.device))
        self.layers.append(layers.ReLU())
        self.layers += model._make_layer(layers.ResNetBottleneck, 64, 3, stride=1).layers
        self.layers += model._make_layer(layers.ResNetBottleneck, 128, 4, stride=2).layers
        self.layers += model._make_layer(layers.ResNetBottleneck, 256, 6, stride=2).layers
        self.layers += model._make_layer(layers.ResNetBottleneck, 512, 3, stride=2).layers
        self.layers.append(layers.AvgPool2D(28))
        self.layers.append(layers.Flatten())
        self.layers.append(layers.Dense(512*layers.ResNetBottleneck.expansion, self.num_classes, device=self.device))
        
        chunk_size, remainder = divmod(len(self.layers), dist.get_world_size())
        self.layers = [self.layers[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(dist.get_world_size())][dist.get_rank()]
        for layer in self.layers:
            layer.init_params()
        
        self.sub_layers = []
        for layer in self.layers:
            layer._get_model_sub_layers(self.sub_layers)
        
        self.streams = [torch.cuda.Stream() for _ in self.sub_layers]
        
        dist.barrier()
        
        if dist.get_rank() == 0:
            x = torch.ones(gmbs, 3, 224, 224, device=self.device)
            for layer in self.layers:
                x = layer(x)
            self.b_recv_buffer = torch.zeros_like(x, device=x.device)
            dist.send(torch.tensor(x.shape, device=x.device), dist.get_rank()+1)
        
        elif dist.get_rank() == dist.get_world_size()-1:
            temp_buffer = torch.zeros([4], device=self.device, dtype=torch.int64)
            dist.recv(temp_buffer, dist.get_rank()-1)
            self.f_recv_buffer = torch.zeros(*temp_buffer, device=self.device)
        
        else:
            temp_buffer = torch.zeros([4], device=self.device, dtype=torch.int64)
            dist.recv(temp_buffer, dist.get_rank()-1)
            self.f_recv_buffer = torch.zeros(*temp_buffer, device=self.device)
            print(self.f_recv_buffer.shape, self.f_recv_buffer.dtype)
            x = torch.ones(*temp_buffer, device=self.device)
            for layer in self.layers:
                x = layer(x)
            self.b_recv_buffer = torch.zeros_like(x)
            dist.send(torch.tensor(x.shape, device=x.device), dist.get_rank()+1)

        self.zero_act()
        
        dist.barrier()
        torch.cuda.synchronize()
        #print(f"Rank {dist.get_rank()}: {self.f_recv_buffer.shape if hasattr(self, 'f_recv_buffer') else ''}, {self.b_recv_buffer.shape if hasattr(self, 'b_recv_buffer') else ''}")
        print(self.layers)
    
class BertBase(Model):
    def __init__(self, device):
        self.device = device

    def init_params(self, input_shape):
        self.layers = []
        num_local_blocks = 12 // dist.get_world_size()
        for _ in range(num_local_blocks):
            layer = layers.BertBlock(768, 12, 3072, layers.GeLU, device=self.device)
            layer.init_params()
            self.layers.append(layer)
        if dist.get_rank() == 0:
            self.layers.insert(0, layers.BertEmbeddings(768, 30522, 512, device=self.device))
            self.layers[0].init_params()
        
        self.f_recv_buffer = torch.zeros(input_shape, device=self.device)
        self.b_recv_buffer = self.f_recv_buffer
        
        self.sub_layers = []
        for layer in self.layers:
            layer._get_model_sub_layers(self.sub_layers)  
        
        self.streams = [torch.cuda.Stream() for _ in self.sub_layers]
        torch.cuda.synchronize()

class BertLarge(Model):
    def __init__(self, device):
        self.device = device

    def init_params(self, input_shape):
        self.layers = []
        num_local_blocks = 24 // dist.get_world_size()
        for _ in range(num_local_blocks):
            layer = layers.BertBlock(1024, 16, 4096, layers.GeLU, device=self.device)
            layer.init_params()
            self.layers.append(layer)
        if dist.get_rank() == 0:
            self.layers.insert(0, layers.BertEmbeddings(1024, 30522, 512, device=self.device))
            self.layers[0].init_params()
        
        self.f_recv_buffer = torch.zeros(input_shape, device=self.device)
        self.b_recv_buffer = self.f_recv_buffer
        
        self.sub_layers = []
        for layer in self.layers:
            layer._get_model_sub_layers(self.sub_layers)  
        
        self.streams = [torch.cuda.Stream() for _ in self.sub_layers]
        torch.cuda.synchronize()
    
        
        
        
            
            
        