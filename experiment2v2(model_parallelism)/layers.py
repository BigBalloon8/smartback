import torch
from torch import vmap as vmap
import numpy as np

from abc import ABC, abstractmethod

class Layer(ABC):
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def initial_pass(self):
        pass
        
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def backward_p1(self):
        pass
    
    @abstractmethod
    def backward_p2(self):
        #p2 is for calculating param grads if theres no params can just pass
        pass
    


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = torch.randn(input_size, output_size)
        self.bias = torch.randn(output_size)
        self._params = {"weights":self.weights, "bias": self.bias}
        
        self.weights_g = torch.zeros_like(self.weights)
        self.bias_g = torch.zeros_like(self.bias)
        self._grads = {"weights":self.weights_g, "bias": self.bias_g}
    
    def initial_pass(self, x):
        self.inputs = torch.zeros_like(x)
        x = vmap(lambda x_: torch.mm(x_, self.weights))(x)
        out = torch.add(x, self.bias)
        self.out = torch.zeros_like(out)
        self.dL_dout = torch.zeros_like(out)
        return out
    
    def forward(self, x):
        self.inputs[:] = x
        x =  vmap(lambda x_: torch.mm(x_, self.weights))(x)
        torch.add(x, self.bias, out=self.out)
        return self.out
    
    def backward_p1(self, dL_dout):
        self.dL_dout[:] = dL_dout
        return vmap(lambda dl_dout: torch.mm(dl_dout, self.weights.T))(dL_dout)
    
    def backward_p2(self):
        self.bias_g = torch.sum(self.dL_dout, dim=0)
        self.weights_g[:] = torch.sum(
            torch.bmm(self.inputs.unsqueeze(2), 
                     self.dL_dout.unsqueeze(1)),
            dim=0)


class MultiHeadAttention(Layer):
    def __init__(self, seq_dim, num_heads):
        self.num_heads = num_heads
        self.seq_dim = seq_dim
        self.linears = {"Q": Dense(self.seq_dim, self.seq_dim),
                        "K": Dense(self.seq_dim, self.seq_dim),
                        "V": Dense(self.seq_dim, self.seq_dim),
                        "O": Dense(self.seq_dim, self.seq_dim)
                        }
    
    def initial_pass(self, Q, K, V, mask=None):
        #TODO change for multihead
        if "cuda" in str(Q.device):
            self.device = "cuda"
            self.streams = []
            for _ in range(len(self.linears)):
                self.streams.append(torch.cuda.Stream())
        else:
            self.device = "cpu"
        
        self.inputs = {"Q": torch.zeros_like(Q),
                       "K": torch.zeros_like(K),
                       "V": torch.zeros_like(V),
        }
        
        lQ = self.linears["Q"].initial_pass(Q)
        lK = self.linears["K"].initial_pass(K)
        lV = self.linears["V"].initial_pass(V)
    
        lQ = torch.cat(lQ.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lK = torch.cat(lK.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lV = torch.cat(lV.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
        QK_T = vmap(lambda q, k: torch.bmm(q, torch.transpose(k, -1, -2)), in_dims=-2, out_dims=-2)(lQ, lK)
        if mask:
            QK_T = vmap(lambda x: x + mask, in_dims=-2, out_dims=-2)(QK_T)
        sQK_T = torch.softmax(QK_T, dim=-1)
        
        self.sQK_T = torch.zeros_like(sQK_T)
        out = vmap(lambda qk_t, v: torch.bmm(qk_t, v), in_dims=-2, out_dims=-2)(sQK_T, lV)
        
        out = torch.flatten(out, -2, -1)
        return self.linears["O"].initial_pass(out)

    
    def forward(self, Q, K, V, mask = None):
        self.inputs["Q"][:] = Q
        self.inputs["K"][:] = K
        self.inputs["V"][:] = V
        
        lQ = self.linears["Q"](Q)
        lK = self.linears["K"](K)
        lV = self.linears["V"](V)
        
        lQ = torch.cat(lQ.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lK = torch.cat(lK.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lV = torch.cat(lV.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
        QK_T = vmap(lambda q, k: torch.bmm(q, torch.transpose(k, -1, -2)), in_dims=-2, out_dims=-2)(lQ, lK)
        if mask:
            QK_T = vmap(lambda x: x + mask, in_dims=-2, out_dims=-2)(QK_T)
        torch.softmax(QK_T, dim=-1, out=self.sQK_T) #TODO add inplace update
        
        out = vmap(lambda qk_t, v: torch.bmm(qk_t, v), in_dims=-2, out_dims=-2)(self.sQK_T, lV)
        out = torch.flatten(out, -2, -1)
        return self.linears["O"](out)

    
    @staticmethod
    def _softmax_jacobian(softmax_out: torch.Tensor): #softmax_out.shape -> N | 1xN
        """
        softmax_out = torch.squeeze(softmax_out)
        jac_base = torch.arange(softmax_out.shape[-1]**2, dtype=torch.int)
        def _jacobian(index):
            row = index // softmax_out.shape[-1]
            col = index - row*softmax_out.shape[-1]
            #print(row, col)
            if row != col:
                return -softmax_out[row]*softmax_out[col]
            else:
                return softmax_out[row]*(1-softmax_out[col])
        jac_base = jac_base.apply_(_jacobian)  # TODO find a better way of doing this
        return jac_base.reshape((softmax_out.shape[-1],)*2)
        """
        softmax_out = torch.squeeze(softmax_out)
        n = softmax_out.shape[-1]
        
        jac_base = -softmax_out.view(n, 1) * softmax_out.view(1, n)
        diag = softmax_out*(1-softmax_out)
        jac_base[torch.arange(n), torch.arange(n)] = diag
        
        return jac_base

        
    def backward_p1(self, dL_dout):
        dL_dAtt = self.linears["O"].backward_p1(dL_dout)
        
        dL_dAtt = torch.cat(dL_dAtt.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
        lV = torch.cat(self.linears["V"].out.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        dL_dsQKT = vmap(lambda dl_dout, v: torch.bmm(dl_dout, torch.transpose(v, -1,-2)), in_dims=-2, out_dims=-2)(dL_dAtt, lV)
        
        # vmap across 3 dims BxCxH 
        J_sQKT = vmap(vmap(vmap(self._softmax_jacobian)))(self.sQK_T)  # sQK_T.shape -> BxCxHxC 
        dL_dQKT = torch.squeeze(vmap(vmap(vmap(torch.mm)))(dL_dsQKT.unsqueeze(-2), J_sQKT))
        
        lK = torch.cat(self.linears["K"].out.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lQ = torch.cat(self.linears["Q"].out.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
        print(dL_dAtt.shape, lV.shape)
        # TODO verifiy this section
        dL_dlQ = vmap(lambda dl_dqkt, k: torch.bmm(dl_dqkt, k), in_dims=-2, out_dims=-2)(dL_dQKT, lK)  # k.T not necessary as its k.T.T  
        dL_dlKT = vmap(lambda dl_dqkt, q: torch.bmm(torch.transpose(q, -1,-2), dl_dqkt), in_dims=-2, out_dims=-2)(dL_dQKT, lQ)  
        dL_dlV = vmap(lambda dl_datt, v: torch.bmm(torch.transpose(v, -1,-2), dl_datt), in_dims=-2, out_dims=-2)(dL_dAtt, lV)  # TODO fix thsi to get correct shape
         
        print(dL_dlQ.shape, dL_dlKT.shape, dL_dlV.shape)
        dL_dQ = self.linears["Q"].backward_p1(torch.flatten(dL_dlQ, -2, -1))
        dL_dK = self.linears["K"].backward_p1(torch.flatten(torch.vmap(lambda dl_dlkt: torch.transpose(dl_dlkt, -1, -2), in_dims=-2, out_dims=-2)(dL_dlKT), -2, -1))
        dL_dV = self.linears["V"].backward_p1(torch.flatten(dL_dlV, -2, -1))
        
        return dL_dQ, dL_dK, dL_dV
        
        
    def backward_p2(self):
        if self.device == "cuda":
            for k, s in zip(self.linears.keys(), self.streams):
                with torch.cuda.stream(s):
                    self.linears[k].backward_p2()
        else:
            for k in self.linears.keys():
                self.linears[k].backward_p2()
                    
    
        
if __name__ == "__main__":
    layer = MultiHeadAttention(80, 8)
    x = torch.randn(16, 24, 80)
    dL_dout = torch.randn(16, 24, 80)
    print(layer.initial_pass(x,x,x).shape)
    layer.forward(x,x,x)
    layer.backward_p1(dL_dout)
        
        
    