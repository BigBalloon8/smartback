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
        self.dL_dout = torch.zeros_like(out)
        return out
    
    def forward(self, x):
        self.inputs[:] = x
        x =  vmap(lambda x_: torch.mm(x_, self.weights))(x)
        out = torch.add(x, self.bias)
        return out
    
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
        return  torch.maximum(x, torch.tensor(0.0, dtype=x.dtype))
    
    def forward(self, x):
        self.inputs[:] = x
        out = torch.maximum(x, torch.tensor(0.0, dtype=x.dtype))
        return out
    
    def backward_p1(self, dL_dout):
        dout_din = torch.where(self.inputs>0, 1.0, 0.0)
        return dL_dout*dout_din

    def backward_p2(self):
        pass
    
class Dropout(Layer):
    def __init__(self, p=0.1):
        self.p = p
        self.training = True
    
    def initial_pass(self, x):
        self.p_mask = torch.bernoulli(torch.ones_like(x) - self.p)
        return x*self.p_mask
    
    def forward(self, x):
        if not self.training or self.p==0:
            return x
        if self.p==1:
            return torch.zeros_like(x)
        self.p_mask[:] = torch.bernoulli(torch.ones_like(x) -self.p)
        return x*self.p_mask
    
    def backward_p1(self, dL_dout):
        if self.p==0:
            return dL_dout
        if self.p==1:
            return torch.zeros_like(dL_dout)
        return dL_dout*self.p_mask
    
    def backward_p2(self):
        pass

    
class MultiHeadAttention(Layer):
    #TODO self is the first layer in the model you can cache the linear outputs of the first 3 linears
    def __init__(self, seq_dim, num_heads, p=0.1):
        self.num_heads = num_heads
        self.seq_dim = seq_dim
        self.linears = {"Q": Dense(self.seq_dim, self.seq_dim),
                        "K": Dense(self.seq_dim, self.seq_dim),
                        "V": Dense(self.seq_dim, self.seq_dim),
                        "O": Dense(self.seq_dim, self.seq_dim)
                        }
        self.dropouts = {"Q": Dropout(p=p),
                         "K": Dropout(p=p),
                         "V": Dropout(p=p)
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
        
        lQ = self.dropouts["Q"].initial_pass(self.linears["Q"].initial_pass(Q))
        lK = self.dropouts["K"].initial_pass(self.linears["K"].initial_pass(K))
        lV = self.dropouts["V"].initial_pass(self.linears["V"].initial_pass(V))
        
        self.lQ = torch.zeros_like(lQ)
        self.lK = torch.zeros_like(lK)
        self.lV = torch.zeros_like(lV)
    
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
        
        self.lQ[:] = self.dropouts["Q"](self.linears["Q"](Q))
        self.lK[:] = self.dropouts["K"](self.linears["K"](K))
        self.lV[:] = self.dropouts["V"](self.linears["V"](V))
        
        lQ = torch.cat(self.lQ.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lK = torch.cat(self.lK.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lV = torch.cat(self.lV.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
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
        
        lV = torch.cat(self.lV.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        dL_dsQKT = vmap(lambda dl_dout, v: torch.bmm(dl_dout, torch.transpose(v, -1,-2)), in_dims=-2, out_dims=-2)(dL_dAtt, lV)
        
        # vmap across 3 dims BxCxH 
        J_sQKT = vmap(vmap(vmap(self._softmax_jacobian)))(self.sQK_T)  # sQK_T.shape -> BxCxHxC 
        dL_dQKT = torch.squeeze(vmap(vmap(vmap(torch.mm)))(dL_dsQKT.unsqueeze(-2), J_sQKT))
        
        lK = torch.cat(self.lK.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lQ = torch.cat(self.lQ.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
        # TODO verifiy this section
        dL_dlQ = vmap(lambda dl_dqkt, k: torch.bmm(dl_dqkt, k), in_dims=-2, out_dims=-2)(dL_dQKT, lK)  # k.T not necessary as its k.T.T  
        dL_dlKT = vmap(lambda dl_dqkt, q: torch.bmm(torch.transpose(q, -1,-2), dl_dqkt), in_dims=-2, out_dims=-2)(dL_dQKT, lQ)  
        dL_dlV = vmap(lambda dl_datt, sqkt: torch.bmm(torch.transpose(sqkt, -2, -1), dl_datt), in_dims=-2, out_dims=-2)(dL_dAtt, self.sQK_T) 
        
        dL_dQ = self.linears["Q"].backward_p1(self.dropouts["Q"].backward_p1(torch.flatten(dL_dlQ, -2, -1)))
        dL_dK = self.linears["K"].backward_p1(self.dropouts["K"].backward_p1(torch.flatten(torch.vmap(lambda dl_dlkt: torch.transpose(dl_dlkt, -1, -2), in_dims=-2, out_dims=-2)(dL_dlKT), -2, -1)))
        dL_dV = self.linears["V"].backward_p1(self.dropouts["V"].backward_p1(torch.flatten(dL_dlV, -2, -1)))
        
        return dL_dQ, dL_dK, dL_dV
        
        
    def backward_p2(self): 
        #TODO add sychronize streams within model
        if self.device == "cuda":
            for k, s in zip(self.linears.keys(), self.streams):
                with torch.cuda.stream(s):
                    self.linears[k].backward_p2()
        else:
            for k in self.linears.keys():
                self.linears[k].backward_p2()
                    
class NLPLayerNorm(Layer):
    def __init__(self, dim, dim_size, eps=1e-08):
        self.dim = dim #seqdim for nlp
        self.dim_size = dim
        self.eps = eps
        
        self.gamma = torch.randn(dim_size)
        self.bias = torch.zeros(dim_size)
        
        self.gamma_g = torch.zeros(dim_size)
        self.bias_g = torch.zeros(dim_size)
    
    def initial_pass(self, x):
        mean = torch.mean(x, dim=self.dim, keepdim=True)
        x_sub_mean = x-mean
        self.x_sub_mean = torch.zeros_like(x_sub_mean)
        var = torch.mean(torch.square(x_sub_mean), dim=self.dim, keepdim=True) + self.eps
        self.var = torch.zeros_like(var)
        norm_x = (x_sub_mean)/torch.sqrt(var)
        self.norm_x = torch.zeros_like(norm_x)
        self.dL_dout = torch.zeros_like(norm_x)
        return norm_x*self.gamma + self.bias
        
    def forward(self, x):
        mean = torch.mean(x, dim=self.dim, keepdim=True)
        self.x_sub_mean[:] = x-mean
        self.var[:] = torch.mean(torch.square(self.x_sub_mean), dim=self.dim, keepdim=True) + self.eps
        self.norm_x[:] = (self.x_sub_mean)/torch.sqrt(self.var)
        return self.norm_x*self.gamma + self.bias
    
    def _norm_jacobian(self):
        # F_Jij is pass vectors x,z and scalars u,v of vector x
        def _F_Jij(x, z, v):
            const_n2 = self.dim_size**2
            f = lambda __x, __z, _v, g: g*((-torch.sqrt(_v)/self.dim_size)-__z*((__x)/const_n2))/_v
            def i_for_j(_x, _z):
                return vmap(f, in_dims=(None, 0, None, 0))(_x, _z, v, self.gamma)
            return vmap(i_for_j, in_dims=(0,None))(x, z)

        def _F_Jii(x, z, v):
            const_n2 = self.dim_size**2
            f = lambda __x, __z, _v, g: g*(((1-1/self.dim_size)*torch.sqrt(_v))-__z*((__x)/const_n2))/_v
            return vmap(f, in_dims=(0,0,None,0))(x, z, v, self.gamma)

        def _diag_set(jac, _diag):
            jac.diagonal()[:]= _diag
            return jac
        
        jac_base = vmap(vmap(_F_Jij))(self.x_sub_mean, self.norm_x,  self.var).squeeze()
        diag = vmap(vmap(_F_Jii))(self.x_sub_mean, self.norm_x, self.var).squeeze()
        return vmap(vmap(_diag_set))(jac_base, diag)
        
        
    def backward_p1(self, dL_dout):
        self.dL_dout[:] = dL_dout
        J = self._norm_jacobian()
        return vmap(vmap(torch.mm))(dL_dout.unsqueeze(-2), J).squeeze()

        
    def backward_p2(self):
        self.bias_g[:] = torch.sum(dL_dout, dim=tuple(range(self.dL_dout.ndim)[:-1]))
        self.gamma_g[:] = torch.sum(dL_dout*self.norm_x, dim=tuple(range(self.dL_dout.ndim)[:-1]))
      
        
class NLPRMSNorm(Layer):
    def __init__(self) -> None:
        super().__init__()


        
class BertBlock(Layer):
    def __init__(self, seq_dim, num_heads, dim_ff, activation=Relu, eps=1e-05, p=0.1):
        self.multihead = MultiHeadAttention(seq_dim=seq_dim, num_heads=num_heads)
        self.linears = {0: Dense(seq_dim, dim_ff),
                        1: Dense(dim_ff, seq_dim)}
        self.ff_act = activation()
        self.norms = {"multi_head": NLPLayerNorm(-1, seq_dim, eps=eps),
                      "ff": NLPLayerNorm(-1, seq_dim, eps=eps)}
        self.dropouts = {"multi_head": Dropout(p=p),
                         "ff": Dropout(p=p)}
    
    def initial_pass(self, x):
        if "cuda" in str(x.device):
            self.device = "cuda"
            self.streams = []
            for _ in range(5):  # 5 Layers with parameters
                self.streams.append(torch.cuda.Stream())
        else:
            self.device = "cpu"

        mh_out = self.multihead.initial_pass(x, x, x) + x
        norm_mh_out = self.norms["multi_head"].initial_pass(mh_out)
        norm_mh_out = self.dropouts["multi_head"].initial_pass(norm_mh_out)
        
        ff1 = self.linears[0].initial_pass(norm_mh_out)
        a = self.ff_act.initial_pass(ff1)
        ff2 = self.linears[1].initial_pass(a) + norm_mh_out
        ff2_norm = self.norms["ff"].initial_pass(ff2)
        return self.dropouts["ff"].initial_pass(ff2_norm)
        
    def forward(self, x):
        mh_out = self.multihead(x, x, x) + x
        norm_mh_out = self.norms["multi_head"](mh_out)
        norm_mh_out = self.dropouts["multi_head"](norm_mh_out)
        
        ff1 = self.linears[0](norm_mh_out)
        a = self.ff_act(ff1)
        ff2 = self.linears[1](a) + norm_mh_out
        ff2_norm = self.norms["ff"](ff2)
        return self.dropouts["ff"](ff2_norm)
    
    def backward_p1(self, dL_dout):
        dL_dff2norm = self.dropouts["ff"].backward_p1(dL_dout)
        dL_dff2 = self.norms["ff"].backward_p1(dL_dff2norm)
        dL_da = self.linears[1].backward_p1(dL_dff2)
        dL_dff1 = self.ff_act.backward_p1(dL_da)
        dL_dnormmhout = self.linears[0].backward_p1(dL_dff1) + dL_dout
        
        dL_dnormmhout = self.dropouts["multi_head"].backward_p1(dL_dnormmhout)
        dL_dmhout = self.norms["multi_head"].backward_p1(dL_dnormmhout)
        dL_din1 = torch.sum(torch.stack(self.multihead.backward_p1(dL_dmhout)),dim=0)
        return dL_din1 + dL_dmhout  # dLdmhout == dL_din2
    
    def backward_p2(self):
        if "cuda" in str(x.device):
            with torch.cuda.stream(self.streams[0]):
                self.multihead.backward_p2()
                
            for i, s in zip(range(2), self.streams[1:3]):
                with torch.cuda.stream(s):
                    self.linears[i].backward_p2()
            
            for k, s in zip(self.norms.keys(), self.streams[3:]):
                with torch.cuda.stream(s):
                    self.norms[k].backward_p2()
        else:
            self.multihead.backward_p2()
            for i in range(2):
                self.linears[i].backward_p2()
            for k in self.norms.keys():
                self.norms[k].backward_p2()
            

        
if __name__ == "__main__":
    x = torch.randn(16, 24, 80)
    dL_dout = torch.randn(16, 24, 80)
    layer = BertBlock(80, 8, 160)
    layer.initial_pass(x)
    layer.forward(x)
    layer.backward_p1(dL_dout)
    layer.backward_p2()
        
        
    