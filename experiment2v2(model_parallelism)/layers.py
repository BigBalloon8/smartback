import torch
from torch import vmap as vmap
import numpy as np

from abc import ABC, abstractmethod

class Layer(ABC):
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
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
    
    @property
    def params(self):
        if hasattr(self, "_params"):
            return self._params
        else:
            raise AttributeError(f"{type(self)} Layer has no params")
    
    def grads(self):
        if hasattr(self, "_grads"):
            return self._grads
        else:
            raise AttributeError(f"{type(self)} Layer has no grads")


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = torch.randn(input_size, output_size)
        self.bias = torch.randn(output_size)
        self._params = {"weights":self.weights, "bias": self.bias}
        
        self.weights_g = torch.zeros_like(self.weights)
        self.bias_g = torch.zeros_like(self.bias)
        self._grads = {"weights":self.weights_g, "bias": self.bias_g}
    
    def initial_pass(self, x):
        self.input = torch.zeros_like(x)
        out = self.forward(x)
        self.out = torch.zeros_like(out)
        self.dL_dout = torch.zeros_like(out)
        return out
    
    def forward(self, x):
        self.inputs[:] = x
        x = torch.mm(x, self.weights)
        torch.add(x, self.bias, out=self.out)
        return self.out
    
    def backward_p1(self, dL_dout):
        self.dL_dout[:] = dL_dout
        return torch.mm(dL_dout, self.weights.T)
    
    def backward_p2(self):
        self.bias_g= torch.sum(self.dL_dout, dim=0)
        self.weights_g["weights"][:] = torch.sum(
            torch.einsum('bij,bjk->bik', 
                         self.inputs.unsqueeze(2), 
                         self.dL_dout.unsqueeze(1)),
            dim=0
        )
        
class Softmax(Layer):
    def initial_pass(self, x, dim=-1):
        self.input = torch.zeros_like(x)
        out = self.forward(x)
        self.out = torch.zeros_like(out)
        self.dL_dout = torch.zeros_like(out)
        return out
    
        

class MultiHeadAttention(Layer):
    def __init__(self, seq_dim, num_heads):
        self.num_heads = num_heads
        self.seq_dim = seq_dim
        self.linears = {"Q": Dense(self.seq_dim, self.seq_dim),
                        "K": Dense(self.seq_dim, self.seq_dim),
                        "V": Dense(self.seq_dim, self.seq_dim)}
    
    def initial_pass(self, Q, K, V, mask):
        #TODO change for multihead
        self.inputs = {"Q": torch.zeros_like(Q),
                       "K": torch.zeros_like(K),
                       "V": torch.zeros_like(V)}
        
        out = self.forward(Q, K, V, mask)
        self.sQK_T = torch.zeros(1) # TODO update this
        self.out = torch.zeros_like(out)
        self.dL_dout = torch.zeros_like(out)
        return out
    
    def forward(self, Q, K, V, mask):
        self.inputs["Q"][:] = Q
        self.inputs["K"][:] = K
        self.inputs["V"][:] = V
        
        lQ = self.linears["Q"](Q)
        lK = self.linears["K"](K)
        lV = self.linears["V"](V)
        
        lQ = torch.cat(lQ.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lK = torch.cat(lK.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lV = torch.cat(lV.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
        QK_T = vmap(lambda q, k: torch.mm(q, torch.transpose(k, -1, -2)), in_dims=-2, out_dims=-2)(Q, K)
        QK_T = vmap(lambda x: x + mask, in_dims=-2, out_dims=-2)(QK_T)
        torch.softmax(QK_T, dim=-1, out=self.sQK_T) #TODO add inplace update
        
        out = vmap(lambda qk_t, v: torch.mm(qk_t, v), in_dims=-2, out_dims=-2)(self.sQK_T, V)
        self.out[:] = out.reshape(out.shape[:2], -1)
        return self.out
    
    @staticmethod
    def _softmax_jacobian(softmax_out: torch.Tensor): #softmax_out.shape -> N | 1xN
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
        
        jac_base = np.vectorize(_jacobian)(jac_base) #TODO find a better way of doing this
        return jac_base.reshape((softmax_out.shape[-1],)*2)

        
    def backward_p1(self, dL_dout):
        self.dL_dout[:] = dL_dout
        
        dL_dout = torch.cat(dL_dout.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
        lV = torch.cat(self.linears["V"].out.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        dL_dsQKT = vmap(lambda dl_dout, v: torch.mm(dl_dout, torch.transpose(v, -1,-2)), in_dims=-2, out_dims=-2)(dL_dout, lV)
        
        # vmap across 3 dims BxCxH 
        J_sQKT = vmap(vmap(vmap(self._softmax_jacobian)))(self.sQK_T)  # sQK_T.shape -> BxCxHxC 
        dL_dQKT = vmap(vmap(vmap(torch.mm)))(dL_dsQKT, J_sQKT)
        
        dL_dlQ = ...
        dL_dlK = ...
        
        dL_dQ = self.linears["Q"].backward_p1(dL_dlQ)
        dL_dK = self.linears["K"].backward_p1(dL_dlK)
        dL_dV = self.linears["V"].backward_p1(dL_dout)
        
        return dL_dQ, dL_dK, dL_dV
        
        
    def backward_p2(self):
        return super().backward_p2()    
    
        
if __name__ == "__main__":
    layer = MultiHeadAttention(10, 8)
    b = torch.softmax(torch.randn(10), dim=-1)
    print(layer._softmax_jacobian(b))
        
        
    