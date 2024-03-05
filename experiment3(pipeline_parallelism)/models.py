from typing import Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist

import layers
from loss import Loss

@contextmanager
def nvtx_profile(name):
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(name)
    yield
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

class Model:
    def __init__(self):
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.rank_rep_0 = 0  # for chimera
        self.rank_rep_1 = 0  # for chimera
        self.layers: list[layers.Layer] = []
        self._2bp = True
        self.criterion: Loss = ...
        self.pipe_algo = "none"
        self.x = ...
        self.dL_dout = ...
        self.f_recv_buffer = ...
        self.b_recv_buffer = ...
        self.sub_layers: list[layers.Layer] = []
        self.streams: list[torch.cuda.Stream] = []
    
    def _rank_0_fwd(self, x:torch.Tensor):
        with nvtx_profile(f"Rank {dist.get_rank()} fwd"):
            for layer in self.layers:
                x = layer(x)
            dist.send(x.contiguous(), self.rank+1)
    
    def _rank_n_fwd(self):
        dist.recv(self.f_recv_buffer, self.rank-1)
        x = self.f_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} fwd"):
            for layer in self.layers:
                x = layer(x)
            dist.send(x.contiguous(), self.rank+1)
    
    def _rank_N_fwd(self, y):
        dist.recv(self.f_recv_buffer, self.rank-1)
        with nvtx_profile(f"Rank {dist.get_rank()} fwd"):
            x = self.f_recv_buffer.clone()
            for layer in self.layers:
                x = layer(x)
            return self.criterion(x, y)
    
    def _rank_0_bwd_p1(self, step=0):
        dist.recv(self.b_recv_buffer, self.rank+1)
        with nvtx_profile(f"Rank {dist.get_rank()} bwd"):
            dL_dout = self.b_recv_buffer.clone()
            for layer in self.layers[::-1]:
                dL_dout = layer.backward_p1(dL_dout, step)
    
    def _rank_n_bwd_p1(self, step=0):
        dist.recv(self.b_recv_buffer, self.rank+1)
        with nvtx_profile(f"Rank {dist.get_rank()} bwd"):
            dL_dout = self.b_recv_buffer.clone()
            for layer in self.layers[::-1]:
                dL_dout = layer.backward_p1(dL_dout, step)
            dist.send(dL_dout.contiguous(), self.rank-1)
    
    def _rank_N_bwd_p1(self, step=0):
        with nvtx_profile(f"Rank {dist.get_rank()} bwd"):
            dL_dout = self.criterion.backward()
            for layer in self.layers[::-1]:
                dL_dout = layer.backward_p1(dL_dout, step)
            dist.send(dL_dout.contiguous(), self.rank-1)
    
    def _backward_p2(self):
        with nvtx_profile(f"Rank {dist.get_rank()} bwd_p2"):
            if self.device == "cuda":
                for i, layer in enumerate(self.sub_layers):
                    with torch.cuda.stream(self.streams[i]):
                        layer.backward_p2()
                        self.streams[i].synchronize()
                        
            else:
                for layer in self.layers:
                    layer.backward_p2()

    def train_step(self, x, y):
        if dist.get_world_size() == 1:
            for layer in self.layers:
                x = layer(x)
            loss = self.criterion(x, y)
            dl_dlout = self.criterion.backward()
            for layer in self.layers[::-1]:
                dl_dlout = layer.backward_p1(dl_dlout)
            if self._2bp:
                self._backward_p2()
            return [loss]

        losses = []
        if self.pipe_algo == "none":
            if self.rank == 0:
                self._rank_0_fwd(x)
                self._rank_0_bwd_p1()
            elif self.rank == self.size - 1:
                loss = self._rank_N_fwd(y)
                self._rank_N_bwd_p1()
                losses.append(loss)
            else:
                self._rank_n_fwd()
                self._rank_n_bwd_p1()
            if self._2bp:
                self._backward_p2()
        
        elif self.pipe_algo == "gpipe":
            if self.rank == 0:
                micro_batches = torch.chunk(x, self.size)
            if self.rank == self.size - 1:
                micro_batches = torch.chunk(y, self.size)
            step = range(self.size)
            
            if self.rank == 0:
                for mb in micro_batches:
                    self._rank_0_fwd(mb)
                for _ in step:
                    self._rank_0_bwd_p1()
                
            elif self.rank == self.size - 1:
                for mb in micro_batches:
                    loss = self._rank_N_fwd(mb)
                for _ in step:
                    self._rank_N_bwd_p1()
                losses.append(loss)

            else:
                for _ in step:
                    self._rank_n_fwd()
                for _ in step:
                    self._rank_n_bwd_p1()
            
            if self._2bp:
                self._backward_p2()
        
        
        elif self.pipe_algo == "1f1b-2":
            if self.rank == 0:
                micro_batches = list(torch.chunk(x, self.size*2))
            elif self.rank == self.size - 1:
                micro_batches = list(torch.chunk(y, self.size*2))
            else:
                micro_batches = list(range(self.size*2))
            # Warmup
            for _ in range(self.size - self.rank):
                if self.rank == 0:
                    self._rank_0_fwd(micro_batches.pop(0))
                elif self.rank == self.size - 1:
                    loss = self._rank_N_fwd(micro_batches.pop(0))
                    losses.append(loss)
                else:
                    micro_batches.pop()
                    self._rank_n_fwd()
            #Full Overlap
            while len(micro_batches) > 0:
                if self.rank == 0:
                    self._rank_0_bwd_p1()
                    self._rank_0_fwd(micro_batches.pop(0))
                    
                elif self.rank == self.size - 1:
                    self._rank_N_bwd_p1()
                    loss = self._rank_N_fwd(micro_batches.pop(0))
                    losses.append(loss)
                else:
                    micro_batches.pop()
                    self._rank_n_bwd_p1()
                    self._rank_n_fwd()
            # Cool Down
            for _ in range(self.size - self.rank):
                if self.rank == 0:
                    self._rank_0_bwd_p1()
                elif self.rank == self.size - 1:
                    self._rank_N_bwd_p1()
                else:
                    self._rank_n_bwd_p1()
        
            if self._2bp:
                self._backward_p2()
        

        elif self.pipe_algo == "1f1b-1":
            if self.rank == 0:
                micro_batches = list(torch.chunk(x, self.size))
            elif self.rank == self.size - 1:
                micro_batches = list(torch.chunk(y, self.size))
            else:
                micro_batches = list(range(self.size))
            # Warmup
            for _ in range(self.size - self.rank):
                if self.rank == 0:
                    self._rank_0_fwd(micro_batches.pop(0))
                elif self.rank == self.size - 1:
                    loss = self._rank_N_fwd(micro_batches.pop(0))
                    losses.append(loss)
                else:
                    micro_batches.pop()
                    self._rank_n_fwd()
            while len(micro_batches) > 0:
                if self.rank == 0:
                    self._rank_0_bwd_p1()
                    self._rank_0_fwd(micro_batches.pop(0))
                    
                elif self.rank == self.size - 1:
                    self._rank_N_bwd_p1()
                    loss = self._rank_N_fwd(micro_batches.pop(0))
                    losses.append(loss)
                else:
                    micro_batches.pop()
                    self._rank_n_bwd_p1()
                    self._rank_n_fwd()
            for _ in range(self.size - self.rank):
                if self.rank == 0:
                    self._rank_0_bwd_p1()
                elif self.rank == self.size - 1:
                    self._rank_N_bwd_p1()
                else:
                    self._rank_n_bwd_p1()
            
            if self._2bp:
                self._backward_p2()
 
        dist.barrier()
        return losses

    def update(self):
        with nvtx_profile(f"Rank {dist.get_rank()} Update"):
            for layer, s in zip(self.sub_layers, self.streams):
                with torch.cuda.stream(s):
                    layer.update()
        dist.barrier()
    
    def multi_stage(self, _bool):
        for layer in self.layers:
            layer.multi_stage_set(_bool)
        self._2bp = _bool
    
    def zero_grad(self):
        with nvtx_profile(f"Rank {dist.get_rank()} Update"):
            for layer in self.layers:
                layer.zero_grad()
        dist.barrier()
    
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
        tensor = torch.tensor(sum(tensor_sizes), device=self.device).clone()
        dist.all_reduce(tensor)
        return tensor.item()

    
class Transformer(Model):
    def __init__(self, dim_size, num_heads, num_kv_heads, max_seqlen, num_blocks, vocab_size, criterion, pipe_algo="none", device="cuda"):
        super().__init__()
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
        self.criterion = criterion
        self.pipe_algo = pipe_algo
        
    
    def init_params(self, gbs, input_shape):
        self.layers = []
        num_local_blocks = self.num_blocks // dist.get_world_size()
        for _ in range(num_local_blocks):
            layer = layers.TransformerPPBlock(**self.block_kwargs)
            layer.init_params()
            self.layers.append(layer)
        
        if dist.get_rank() == 0:
            self.layers.insert(0, layers.Embeddings(**self.embedding_kwargs))
            self.layers[0].init_params()
        if dist.get_rank() == dist.get_world_size() - 1:
            self.layers.append(layers.NLPRMSNorm(-1, self.block_kwargs["dim"], device=self.block_kwargs["device"]))
            self.layers.append(layers.Dense(self.block_kwargs["dim"], self.embedding_kwargs["num_embeddings"], bias=False, device=self.block_kwargs["device"]))
            self.layers[-1].init_params()
            self.layers[-2].init_params()
        
        if self.pipe_algo in ("gpipe", "1f1b-1"):
            mb_size = (gbs // dist.get_world_size(),)
        elif self.pipe_algo == "1f1b-2":
            mb_size = (gbs // (dist.get_world_size() * 2),)
        else:
            mb_size = (gbs,)

        self.f_recv_buffer = torch.zeros(mb_size + input_shape, device=self.device)
        self.b_recv_buffer = self.f_recv_buffer

        self.sub_layers = []
        for layer in self.layers:
            layer._get_model_sub_layers(self.sub_layers)   
        
        self.streams = [torch.cuda.Stream() for _ in self.sub_layers]
        torch.cuda.synchronize()
            
            

class ResNet50(Model):
    def __init__(self, num_classes, device):
        super().__init__()
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
    
        
        
class Unet(Model):
    
            
            
        