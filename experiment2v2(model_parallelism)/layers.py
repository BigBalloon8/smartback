import torch
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
        
        lQ = torch.cat(lQ.unsqueeze(-2).chunk(num_heads, dim=-1), dim=-2)
        lK = torch.cat(lK.unsqueeze(-2).chunk(num_heads, dim=-1), dim=-2)
        lV = torch.cat(lV.unsqueeze(-2).chunk(num_heads, dim=-1), dim=-2)
        
        QK_T = torch.vmap(lambda q, k: torch.mm(q, torch.transpose(k, -1, -2)), in_dims=-2, out_dims=-2)(Q, K)
        QK_T = torch.vmap(lambda x: x + mask, in_dims=-2, out_dims=-2)(QK_T)
        sQK_T = lambda x: torch.softmax(x, dim=-1)(QK_T) #TODO add inplace update
        
        out = torch.vmap(lambda qk_t, v: torch.mm(qk_t, v), in_dims=-2, out_dims=-2)(sQK_T, V)
        self.out[:] = out.reshape(out.shape[:2], -1)
        return self.out
    
    def backward_p1(self, dL_dout):
        self.dL_dout[:] = dL_dout
        
        dL_dout = torch.cat(dL_dout.unsqueeze(-2).chunk(num_heads, dim=-1), dim=-2)
        
        lV = torch.cat(self.linears["V"].out.unsqueeze(-2).chunk(num_heads, dim=-1), dim=-2)
        dL_dsQKT = torch.vmap(lambda dl_dout, v: torch.mm(dl_dout, torch.transpose(v, -1,-2)), in_dims=-2, out_dims=-2)(dL_dout, lV)
        J_sQKT = 
        
        
        
        
        
        
    