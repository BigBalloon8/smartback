from contextlib import contextmanager
from typing import Self
from termcolor import colored

import torch
import torch.distributed as dist

import layers
from loss import Loss


COLOURS = ["red","green", "yellow", "cyan"]



@contextmanager
def nvtx_profile(name:str):
    
    #torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(name)
    print(colored(name, COLOURS[dist.get_rank()]))
    yield
    #torch.cuda.synchronize()
    print(colored(name.upper(), COLOURS[dist.get_rank()]))
    torch.cuda.nvtx.range_pop()


class CommHandler:
    def __init__(self, model):
        self.model = model
        self.fmb0 = 0
        self.fmb1 = 0
        self.bmb0 = 0
        self.bmb1 = 0
    def forward(self):
        if self.model.rank_rep_0 < self.model.rank_rep_1:
            if self.model.rank_rep_0 == 0:
                dist.send()

    def backward(self):
        ...




class Model:
    def __init__(self):
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.rank_rep_0 = 0  # for chimera
        self.rank_rep_1 = 0  # for chimera
        self.layers: list[list[layers.Layer]] = []
        self._2bp = True
        self.to_send = None
        self.criterion: Loss = ...
        self.pipe_algo = "chimera"
        self.x = ...
        self.dL_dout = ...
        self.f_recv_buffer = ...
        self.b_recv_buffer = ...
        self.sub_layers: list[list[layers.Layer]] = []
        self.streams: list[torch.cuda.Stream] = []


    # Forward
    def _rank_0_rep_0_fwd(self, x, mb):
        with nvtx_profile(f"Rank {dist.get_rank()} fwd {mb}"):
            for layer in self.layers[0]:
                x = layer(x)
        
    
    def _rank_n_rep_0_fwd(self, mb):
        x = self.f_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} fwd {mb}"):
            for layer in self.layers[0]:
                x = layer(x)
        self.to_send = x
    
    def _rank_N_rep_0_fwd(self, y, mb):
        x = self.f_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} fwd {mb}, ---LOSS---"):
            for layer in self.layers[0]:
                x = layer(x)
            loss = self.criterion(x, y)
            return loss
    
    def _rank_0_rep_1_fwd(self, x, mb):
        with nvtx_profile(f"Rank {dist.get_rank()} fwd {mb}"):
            for layer in self.layers[1]:
                x = layer(x)
        self.to_send = x
        #print(f"R1: {self.rank} sent to {self.rank-1}, {mb}")
    
    def _rank_n_rep_1_fwd(self, mb):
        x = self.f_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} fwd {mb}"):
            for layer in self.layers[1]:
                x = layer(x)
        self.to_send = x
    
    def _rank_N_rep_1_fwd(self, y, mb):
        x = self.f_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} fwd {mb}, ---LOSS---"):
            for layer in self.layers[1]:
                x = layer(x)
            loss = self.criterion(x, y)
            return loss
    
    #Backward
    def _rank_0_rep_0_bwd(self, step=0, mb=0):
        dl_dout = self.b_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} bwd {mb}"):
            for layer in reversed(self.layers[0]):
                dl_dout = layer.backward_p1(dl_dout, step)
    
    def _rank_n_rep_0_bwd(self, step=0, mb=0):
        dl_dout = self.b_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} bwd {mb}"):
            for layer in reversed(self.layers[0]):
                dl_dout = layer.backward_p1(dl_dout, step)
        self.to_send = dl_dout
            
    def _rank_N_rep_0_bwd(self, step=0, mb=0):
        with nvtx_profile(f"Rank {dist.get_rank()} bwd {mb}"):
            dl_dout = self.criterion.backward()
            for layer in reversed(self.layers[0]):
                dl_dout = layer.backward_p1(dl_dout, step)
        self.to_send = dl_dout
        
    def _rank_0_rep_1_bwd(self, step=0, mb=0):
        dl_dout = self.b_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} bwd {mb}"):
            for layer in reversed(self.layers[1]):
                dl_dout = layer.backward_p1(dl_dout, step)
    
    def _rank_n_rep_1_bwd(self, step=0, mb=0):
        dl_dout = self.b_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} bwd {mb}"):
            for layer in reversed(self.layers[1]):
                dl_dout = layer.backward_p1(dl_dout, step)
        self.to_send = dl_dout
        
    def _rank_N_rep_1_bwd(self, step=0, mb=0):
        dl_dout = self.b_recv_buffer.clone()
        with nvtx_profile(f"Rank {dist.get_rank()} bwd {mb}"):
            dl_dout = self.criterion.backward()
            for layer in reversed(self.layers[1]):
                dl_dout = layer.backward_p1(dl_dout, step)
        self.to_send = dl_dout
    
    def _backward_p2(self):
        with nvtx_profile(f"Rank {dist.get_rank()} bwd_p2"):
            if self.rank_rep_0 < self.rank_rep_1:
                for i, layer in enumerate(self.sub_layers[0]):
                    with torch.cuda.stream(self.streams[i]):
                        layer.backward_p2()
                        self.streams[i].synchronize()
                    for g in layer.grads.values():
                        dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True)
                for i, layer in enumerate(self.sub_layers[1]):
                    with torch.cuda.stream(self.streams[i]):
                        layer.backward_p2()
                        self.streams[i].synchronize()
                    for g in layer.grads.values():
                        dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True)
            else:
                for i, layer in enumerate(self.sub_layers[1]):
                    with torch.cuda.stream(self.streams[i]):
                        layer.backward_p2()
                        self.streams[i].synchronize()
                    for g in layer.grads.values():
                        dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True)
                for i, layer in enumerate(self.sub_layers[0]):
                    with torch.cuda.stream(self.streams[i]):
                        layer.backward_p2()
                        self.streams[i].synchronize()
                    for g in layer.grads.values():
                        dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True)
                        
    
    def train_step(self, x, y):
        losses = []
        if self.rank_rep_0 == 0 or self.rank_rep_1 == 0:
            x = list(torch.chunk(x, self.size))
        if self.rank_rep_0 == self.size-1 or self.rank_rep_1 == self.size-1:
            y = list(torch.chunk(y, self.size))
        fmb_rep_0 = 0
        bmb_rep_0 = 0
        fmb_rep_1 = self.size // 2
        bmb_rep_1 = self.size // 2
        
        HALF_SIZE = self.size // 2

        if self.rank_rep_0 < self.rank_rep_1:
            # ------------------------------------------------------
            for _ in range(HALF_SIZE - self.rank_rep_0):
                if self.rank_rep_0 == 0:
                    self._rank_0_rep_0_fwd(x.pop(0), fmb_rep_0)
                    if fmb_rep_0 != HALF_SIZE-self.rank_rep_0 -1:
                        dist.send(self.to_send, self.rank + 1)
                else:
                    dist.recv(self.f_recv_buffer, self.rank-1)
                    self._rank_n_rep_0_fwd(fmb_rep_0)
                    if fmb_rep_0 != HALF_SIZE-self.rank_rep_0 -1:
                        dist.send(self.to_send, self.rank + 1)
                fmb_rep_0 += 1
            
            dist.isend(self.to_send, self.rank + 1, tag=fmb_rep_0)

            # ------------------------------------------------------
            for _ in range(self.rank_rep_0):
                
                self._rank_n_rep_1_fwd(fmb_rep_1)
                self._rank_n_rep_0_fwd(fmb_rep_0)
                fmb_rep_0 += 1
                fmb_rep_1 += 1
            
            # ------------------------------------------------------
            for i in range(self.size//2 - self.rank_rep_0):
                if self.rank_rep_0 == 0:
                    self._rank_N_rep_1_fwd(y.pop(0), fmb_rep_1)
                    self._rank_N_rep_1_bwd(step=0, mb=bmb_rep_1)
                else:
                    self._rank_n_rep_1_fwd(fmb_rep_1)
                    self._rank_n_rep_1_bwd(step=0, mb=bmb_rep_1) # Check notes
                fmb_rep_1 += 1
                bmb_rep_1 += 1
            
            # ------------------------------------------------------
            for i in range(self.rank_rep_0):
                self._rank_n_rep_0_bwd(step=0, mb=bmb_rep_0)
                self._rank_n_rep_1_bwd(step=0, mb=bmb_rep_1) # Check Notes
                bmb_rep_0 += 1
                bmb_rep_1 += 1
            
            # ------------------------------------------------------
            for i in range(self.size//2 - self.rank_rep_0):
                if self.rank_rep_0 == 0:
                    self._rank_0_rep_0_bwd(step=0, mb=bmb_rep_0)
                else:
                    self._rank_n_rep_0_bwd(step=0, mb=bmb_rep_0)
                bmb_rep_0 += 1
            
                    
        else:
            # ------------------------------------------------------

            for _ in range(HALF_SIZE - self.rank_rep_1):
                if self.rank_rep_1 == 0:
                    self._rank_0_rep_1_fwd(x.pop(0), fmb_rep_1)
                    if fmb_rep_1 != HALF_SIZE-self.rank_rep_1 -1:
                        dist.send(self.to_send, self.rank - 1)
                else:
                    dist.recv(self.f_recv_buffer, self.rank + 1)
                    self._rank_n_rep_1_fwd(fmb_rep_1)
                    if fmb_rep_1 != HALF_SIZE-self.rank_rep_1 -1:
                        dist.send(self.to_send, self.rank - 1)
                fmb_rep_1 += 1
            
            dist.isend(self.to_send, self.rank - 1, tag=fmb_rep_1)
                
            # ------------------------------------------------------
            for _ in range(self.rank_rep_1):
                self._rank_n_rep_0_fwd(fmb_rep_0)
                self._rank_n_rep_1_fwd(fmb_rep_1)
                fmb_rep_0 += 1
                fmb_rep_1 += 1
            
            # ------------------------------------------------------
            for i in range(self.size//2 - self.rank_rep_1):
                if self.rank_rep_1 == 0:
                    self._rank_N_rep_0_fwd(y.pop(0), fmb_rep_0)
                    self._rank_N_rep_0_bwd(step=0, mb=bmb_rep_0)
                else:
                    self._rank_n_rep_0_fwd(fmb_rep_0)
                    self._rank_n_rep_0_bwd(step=0, mb=bmb_rep_0)
                fmb_rep_0 += 1
                bmb_rep_0 += 1
    
            # ------------------------------------------------------
            for i in range(self.rank_rep_1):
                self._rank_n_rep_1_bwd(step=0,  mb=bmb_rep_1)
                self._rank_n_rep_0_bwd(step=0 , mb=bmb_rep_0) # Check Notes
                bmb_rep_1 += 1
                bmb_rep_0 += 1

            # ------------------------------------------------------
            for i in range(self.size//2 - self.rank_rep_0):
                if self.rank_rep_1 == 0:
                    self._rank_0_rep_1_bwd(step=0, mb=bmb_rep_1)
                else:
                    self._rank_n_rep_1_bwd(step=0, mb=bmb_rep_1)
                bmb_rep_1 += 1

        self._backward_p2()
    
    def update(self):
        with nvtx_profile(f"Rank {dist.get_rank()} update"):
            if not self._2bp:
                if self.rank_rep_0 < self.rank_rep_1:
                    for layer in self.sub_layers[0]:
                        for g in layer.grads.values():
                            dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True)
                    for layer in self.sub_layers[1]:
                        for g in layer.grads.values():
                            dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True)
                else:
                    for layer in self.sub_layers[1]:
                        for g in layer.grads.values():
                            dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True)
                    for layer in self.sub_layers[0]:
                        for g in layer.grads.values():
                            dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True)
            
            torch.cuda.synchronize()
            for i, layer in enumerate(self.sub_layers[0]):
                with torch.cuda.stream(self.streams[i]):
                    layer.update()
            for i, layer in enumerate(self.sub_layers[1]):
                with torch.cuda.stream(self.streams[i]):
                    layer.update()
        dist.barrier()

    def zero_grad(self):
        with nvtx_profile(f"Rank {dist.get_rank()} Zero Grad"):
            for layer in self.sub_layers[0]:
                layer.zero_grad()
            for layer in self.sub_layers[1]:
                layer.zero_grad()

    def multi_stage(self, _bool):
        for layer in self.layers[0]:
            layer.multi_stage_set(_bool)
        for layer in self.layers[1]:
            layer.multi_stage_set(_bool)
        self._2bp = _bool
 
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
        self.layers = [[], []]
        self.rank_rep_0 = dist.get_rank()
        self.rank_rep_1 = dist.get_world_size() - self.rank-1
        #self.mirror_partner = dist.new_group([self.rank_rep_0, self.rank_rep_1])
        num_local_blocks = self.num_blocks // dist.get_world_size()

        torch.manual_seed(self.rank_rep_0)
        torch.cuda.manual_seed(self.rank_rep_0)
        for _ in range(num_local_blocks):
            layer = layers.TransformerPPBlock(**self.block_kwargs)
            layer.init_params()
            self.layers[0].append(layer)
        if self.rank_rep_0 == 0:
            self.layers[0].insert(0, layers.Embeddings(**self.embedding_kwargs))
            self.layers[0][0].init_params()
        if self.rank_rep_0 == dist.get_world_size() - 1:
            self.layers[0].append(layers.NLPRMSNorm(-1, self.block_kwargs["dim"], device=self.block_kwargs["device"]))
            self.layers[0].append(layers.Dense(self.block_kwargs["dim"], self.embedding_kwargs["num_embeddings"], bias=False, device=self.block_kwargs["device"]))
            self.layers[0][-1].init_params()
            self.layers[0][-2].init_params()

        
        torch.manual_seed(self.rank_rep_1)
        torch.cuda.manual_seed(self.rank_rep_1)
        for _ in range(num_local_blocks):
            layer = layers.TransformerPPBlock(**self.block_kwargs)
            layer.init_params()
            self.layers[1].append(layer)
        if self.rank_rep_1 == 0:
            self.layers[1].insert(0, layers.Embeddings(**self.embedding_kwargs))
            self.layers[1][0].init_params()
        if self.rank_rep_1 == dist.get_world_size() - 1:
            self.layers[1].append(layers.NLPRMSNorm(-1, self.block_kwargs["dim"], device=self.block_kwargs["device"]))
            self.layers[1].append(layers.Dense(self.block_kwargs["dim"], self.embedding_kwargs["num_embeddings"], bias=False, device=self.block_kwargs["device"]))
            self.layers[1][-1].init_params()
            self.layers[1][-2].init_params()
        
        mb_size = (gbs//dist.get_world_size(), )
        
        self.f_recv_buffer = torch.zeros(mb_size + input_shape, device=self.device)
        self.b_recv_buffer = self.f_recv_buffer

        self.sub_layers = [[], []]
        for layer in self.layers[0]:
            layer._get_model_sub_layers(self.sub_layers[0])
        for layer in self.layers[1]:
            layer._get_model_sub_layers(self.sub_layers[1])   
        
        self.streams = [[torch.cuda.Stream() for _ in self.sub_layers], [torch.cuda.Stream() for _ in self.sub_layers]]
        torch.cuda.synchronize()