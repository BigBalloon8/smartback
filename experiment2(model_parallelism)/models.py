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

    def forward(self, input):
        if dist.get_rank()  == 0:
            torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Forward")
            for layer in self.layers:
                input = layer(input)
            dist.send(input, dist.get_rank()+1)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            
            
        elif dist.get_rank() == dist.get_world_size()-1:
            dist.recv(self.f_recv_buffer, dist.get_rank()-1)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Forward")
            x = self.f_recv_buffer.clone()
            for layer in self.layers:
                x = layer(x)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            return x
        
        else:
            dist.recv(self.f_recv_buffer, dist.get_rank()-1)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Forward")
            x = self.f_recv_buffer.clone()
            for layer in self.layers:
                x = layer(x)
            dist.send(x, dist.get_rank()+1)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

    def backward(self, dL_dout):
        if dist.get_rank() == dist.get_world_size()-1:
            torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Backward P1")
            for layer in self.layers[::-1]:
                dL_dout = layer.backward_p1(dL_dout)
            torch.cuda.synchronize()
            work = dist.isend(dL_dout, dist.get_rank()-1)
            torch.cuda.nvtx.range_pop()
            if layer.multi_stage:
                torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Backward P2")
                for layer, s in zip(self.sub_layers, self.streams):
                    with torch.cuda.stream(s):
                        layer.backward_p2()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            work.wait()
        
        elif dist.get_rank() == 0:
            dist.recv(self.b_recv_buffer, dist.get_rank()+1)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Backward")
            dL_dout = self.b_recv_buffer.clone()
            for layer in self.layers[::-1]:
                # backward_p2 always called on rank 0 with backward_p1 for memory efficiency 
                dL_dout = layer.backward_p1(dL_dout)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        
        else:
            dist.recv(self.b_recv_buffer, dist.get_rank()+1)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Backward P1")
            dL_dout = self.b_recv_buffer.clone()
            for layer in self.layers[::-1]:
                dL_dout = layer.backward_p1(dL_dout)
            torch.cuda.synchronize()
            dist.isend(dL_dout, dist.get_rank()-1)
            torch.cuda.nvtx.range_pop()
            if layer.multi_stage:
                torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Backward P2")
                for layer, s in zip(self.sub_layers, self.streams):
                    with torch.cuda.stream(s):
                        layer.backward_p2()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        dist.barrier()

    def update(self):
        torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Update Grads")
        for layer in self.layers:
            layer.update()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.nvtx.range_pop()
    
    def multi_stage(self, _bool):
        for layer in self.layers:
            layer.multi_stage_set(_bool)

    def zero_grad(self):
        torch.cuda.nvtx.range_push(f"Rank {dist.get_rank()}: Zero Grad")
        for layer in self.layers:
            layer.zero_grad()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.nvtx.range_pop()

    def save(self, path):
        pass

    def load(self, path):
        pass

    def to(self, arg):
        for layer in self.layers:
            layer.to(arg)
        

    
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
        
    
    def init_params(self, input_shape):
        self.layers = []
        num_local_blocks = self.num_blocks // dist.get_world_size()
        for _ in range(num_local_blocks):
            layer = layers.TransformerPPBlock(**self.block_kwargs)
            layer.init_params()
            self.layers.append(layer)
        if dist.get_rank() == 0:
            self.layers.insert(0, layers.Embeddings(**self.embedding_kwargs))
            self.layers[0].init_params()
        
        self.f_recv_buffer = torch.zeros(input_shape, device=self.device)
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
    
    def init_params(self, gbs):
        model = layers.ResNet(layers.ResNetBottleneck, [3,4,6,3], device=self.device)
        self.layers = []
        self.layers.append(layers.Conv2D(3, 64, 3, stride=1, padding=1, bias=False, device=self.device))
        self.layers.append(layers.BatchNorm2D(64, device=self.device))
        self.layers.append(layers.ReLU())
        self.layers += model._make_layer(layers.ResNetBottleneck, 64, 3, stride=1).layers
        self.layers += model._make_layer(layers.ResNetBottleneck, 128, 4, stride=2).layers
        self.layers += model._make_layer(layers.ResNetBottleneck, 256, 6, stride=2).layers
        self.layers += model._make_layer(layers.ResNetBottleneck, 512, 3, stride=2).layers
        self.layers.append(layers.AvgPool2D(4))
        self.layers.append(layers.Flatten())
        self.layers.append(layers.Dense(512*layers.ResNetBottleneck.expansion, self.num_classes, device=self.device))
        
        chunk_size, remainder = divmod(len(self.layers), dist.get_world_size())
        self.layers = [self.layers[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(dist.get_world_size())][dist.get_rank()]

        for layer in self.layers:
            layer.init_params()
        
        dist.barrier()
        
        if dist.get_rank() == 0:
            x = torch.ones(gbs, 3, 224, 224, device=self.device)
            for layer in self.layers:
                x = layer(x)
            self.b_recv_buffer = torch.zeros_like(x)
            dist.send(torch.tensor(x.shape), dist.get_rank()+1)
        
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
            dist.send(torch.tensor(x.shape), dist.get_rank()+1)
        
        dist.barrier()
        torch.cuda.synchronize()
        print(f"Rank {dist.get_group_rank()}: {self.f_recv_buffer.shape if hasattr(self, 'f_recv_buffer') else ''}, {self.b_recv_buffer.shape if hasattr(self, 'b_recv_buffer') else ''}")
            
            
        