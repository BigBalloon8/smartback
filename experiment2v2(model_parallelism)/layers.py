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
        self.bias_g[:] = torch.sum(self.dL_dout, dim=tuple(range(self.dL_dout.ndim)[:-1]))
        
        
        if self.dL_dout.ndim == 2:
            self.weights_g[:] = torch.sum(
                torch.bmm(self.inputs.unsqueeze(2), 
                            self.dL_dout.unsqueeze(1)
                    ),
            dim=0)
        elif self.dL_dout.ndim == 3:
            self.weights_g[:] = torch.sum(
                torch.bmm(
                    torch.transpose(self.inputs, -2, -1),
                    self.dL_dout
                ),
            dim=tuple(range(self.dL_dout.ndim)[:-2]))
        else:
            raise Exception("ndim of input to Dense not supported")


class Relu(Layer):
    def initial_pass(self, x):
        self.inputs = torch.zeros_like(x)
        return  torch.maximum(x, 0)
    
    def forward(self, x):
        self.inputs[:] = x
        out = torch.maximum(x, 0.0)
        return out
    
    def backward_p1(self, dL_dout):
        dout_din = torch.where(self.inputs>0, 1.0, 0.0)
        return torch.bmm(dL_dout, dout_din)

    def backward_p2(self):
        pass


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
        
        # TODO verifiy this section
        dL_dlQ = vmap(lambda dl_dqkt, k: torch.bmm(dl_dqkt, k), in_dims=-2, out_dims=-2)(dL_dQKT, lK)  # k.T not necessary as its k.T.T  
        dL_dlKT = vmap(lambda dl_dqkt, q: torch.bmm(torch.transpose(q, -1,-2), dl_dqkt), in_dims=-2, out_dims=-2)(dL_dQKT, lQ)  
        dL_dlV = vmap(lambda dl_datt, sqkt: torch.bmm(torch.transpose(sqkt, -2, -1), dl_datt), in_dims=-2, out_dims=-2)(dL_dAtt, self.sQK_T) 
         
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
                    
class BatchNorm(Layer):
    #TODO update for new layer abstract
    def __init__(self, input_size, batch_size, epsilon=1e-5, momentum = 0.99):
        super().__init__(input_size, input_size, batch_size)
        self.inputs = torch.zeros((self.b_size, self.i_size))
        self.out = torch.zeros((self.b_size, self.o_size))
        
        self.mu = torch.zeros(1, self.i_size)
        self.var = torch.ones(1, self.i_size)
        
        self.b_mu = torch.zeros(1, self.i_size)
        self.b_var = torch.zeros(1, self.i_size)
        
        self.eps = epsilon
        self.it_call = 0
        self.momentum = momentum
        
        self.params = {
            "beta": torch.zeros(1, self.i_size),
            "gamma": torch.ones(1, self.i_size)
            } 
        
        self.x_norm = torch.zeros(self.b_size, self.i_size)
        
        
        self.grads = {
            "beta": torch.zeros(1, self.i_size),
            "gamma": torch.zeros(1, self.i_size)
            } 
    
    def forward(self, x):
        self.it_call += 1
        self.inputs[:] = x
        if self.training:
            self.b_mu = torch.mean(x, dim=0).unsqueeze(0)
            self.b_var = torch.var(x, dim=0).unsqueeze(0)
            
            self.x_norm[:] = (x-self.b_mu)/torch.sqrt(self.b_var + self.eps)
            self.out[:] = self.params["gamma"]*self.x_norm + self.params["beta"]
            
            self.mu = self.b_mu * (self.momentum/self.it_call) + \
                            self.mu * (1 - (self.momentum/self.it_call))
            
            self.var = self.b_var * (self.momentum/self.it_call) + \
                        self.var * (1 - (self.momentum/self.it_call))
        
        else:
            self.x_norm[:] = (x-self.mu)/torch.sqrt(self.var + self.epsilon)
            self.out[:] = self.params["gamma"]*self.x_norm + self.params["beta"]
        return self.out
    
    def backward_p1(self, dL_dout):
        if hasattr(self, "dL_dout"):
            self.dL_dout[:] = dL_dout
        else:
            self.dL_dout = dL_dout

        
        #X_mu = self.inputs-self.b_mu
        var_sqrt_inv = 1./torch.sqrt(self.b_var + self.eps)
        
        dout_dXnorm = dL_dout * self.params["gamma"]
        
        return (1./self.b_size) * var_sqrt_inv * (self.b_size*dout_dXnorm - torch.sum(dout_dXnorm, dim=0) - self.x_norm*torch.sum(dout_dXnorm*self.x_norm, dim=0))

    
    def backward_p2(self):
        self.grads["beta"][:] = torch.sum(self.dL_dout, dim=0)
        self.grads["gamma"][:] = torch.sum(self.x_norm*self.dL_dout, dim=0)

class NLPLayerNorm(Layer):
    def __init__(self, dim, dim_size, eps=1e-05):
        self.dim = dim #seqdim for nlp
        self.dim_size = dim
        self.eps = eps
        
        self.gamma = torch.randn(dim_size)
        self.bias = torch.zeros(dim_size)
        
        self.gamma_g = torch.zeros(dim_size)
        self.bias_g = torch.zeros(dim_size)
    
    def forward(self, x):
        self.inputs[:] = x
        self.mean[:] = torch.mean(x, dim=self.dim, keepdim=True)
        self.x_sub_mean[:] = x-self.mean
        self.var[:] = torch.mean(torch.square(self.x_sub_mean), dim=self.dim, keepdim=True) + self.eps
        self.norm_x[:] = (self.x_sub_mean)/torch.sqrt(self.var)
        return self.norm_x*self.gamma + self.bias
    
    def _norm_jacobian(self):
        # F_Jij is pass vectors x,z and scalars u,v of vector x
        def _F_Jij(x, z, u, v):
            const_n2 = self.dim_size**2
            f = lambda __x, __z: ((-torch.sqrt(v)/self.dim_size)-__z*((__x-u)/const_n2))/v
            def i_for_j(_x, _z):
                return vmap(f, in_dims=(None, 0))(_x, _z)
            return vmap(i_for_j, in_dims=(0,None))(x, z)

        def _F_Jii(x, z, u, v):
            const_n2 = self.dim_size**2
            f = lambda __x, __z: (((1-1/self.dim_size)*torch.sqrt(v))-__z*((__x-u)/const_n2))/v
            return vmap(f)(x, z)
            
        def _diag_set(jac, diag):
            jac[torch.arange(self.dim_size), torch.arange(self.dim_size)] = diag
            return jac
        jac_base = vmap(vmap(_F_Jij, in_dims=(0,0,None,None)))(self.inputs, self.norm_x, self.mean, self.var)
        diag = vmap(vmap(_F_Jii, in_dims=(0,0,None,None)))(self.inputs, self.norm_x, self.mean, self.var)
        return vmap(vmap(_diag_set))(jac_base, diag)
        
        
        
    def backward_p1(self, dL_dout):
        self.dL_dout[:] = dL_dout #dL_dout.shape = NxCxE
        
        
class TransformerEncoderBlock(Layer):
    def __init__(self, seq_dim, num_heads, dim_ff, activation=Relu):
        self.multihead = MultiHeadAttention(seq_dim=seq_dim, num_heads=num_heads)
        self.linears = {1: Dense(seq_dim, dim_ff),
                        2: Dense(dim_ff, seq_dim)}
        self.ff_act = activation()
        
        
        
if __name__ == "__main__":
    layer = MultiHeadAttention(80, 8)
    x = torch.randn(16, 24, 80)
    dL_dout = torch.randn(16, 24, 80)
    print(layer.initial_pass(x,x,x).shape)
    layer.forward(x,x,x)
    layer.backward_p1(dL_dout)
    layer.backward_p2()
    # have to fix layer.backward_p2()
        
        
    