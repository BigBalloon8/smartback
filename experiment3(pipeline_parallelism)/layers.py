from typing import Any, Dict, Optional, Union, Sequence, Tuple, List, Final
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps, partial
import math

import torch
import torch.nn.functional as F
from torch import vmap as vmap
import torch.distributed as dist

from fft_conv_pytorch.fft_conv import fft_conv
import selective_scan_cuda
from einops import rearrange, repeat


from wrappers import cleanup_act, multi_stage_wrapper, expose_params
import init_params
import loss

@contextmanager
def nvtx_profile(name):
    yield
    return
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(name)
    yield
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

class Layer(ABC):
    def __init__(self):
        self.multi_stage = True
        self.training = True
        self.params: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}
        self.acts = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self):
        pass
    
    def init_params(self):
        pass

    @abstractmethod
    def backward_p1(self):
        pass
    
    def backward_p2(self, inter=False):
        #p2 is for calculating param grads if theres no params can just pass
        pass

    def update(self):
        pass
    
    def clear_acts(self):
        if self.acts:
            for act in self.acts:
                setattr(self, act, [])
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).clear_acts()
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.clear_acts()
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                         sub_layer.clear_acts()

    def to_(self, arg):
        if hasattr(self, "params"):
            for k in self.params.keys():
                self.params[k] = self.params[k].to(arg)
            for k in self.grads.keys():
                self.grads[k] = self.grads[k].to(arg)
        
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).to_(arg)
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.to_(arg)
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                         sub_layer.to_(arg)
    
    def multi_stage_set(self, _bool):
        self.multi_stage = _bool
        
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).multi_stage_set(_bool)
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.multi_stage_set(_bool)
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                        sub_layer.multi_stage_set(_bool)
    
    def zero_grad(self):
        if self.grads:
            for k in self.grads.keys():
                self.grads[k].zero_()
        
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).zero_grad()
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.zero_grad()
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                        sub_layer.zero_grad()
    
    def _get_model_sub_layers(self, sublayers_list: list):
        if self.grads:
            sublayers_list.append(self)
        
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item)._get_model_sub_layers(sublayers_list)
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer._get_model_sub_layers(sublayers_list)
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                        sub_layer._get_model_sub_layers(sublayers_list)
    
    def get_num_params(self, num_elm_list:list):
        if self.params:
            for param in self.params.values():
                num_elm_list.append(torch.numel(param))
        for item in dir(self):
            if isinstance(getattr(self, item), Layer):
                getattr(self, item).get_num_params(num_elm_list)
            elif isinstance(getattr(self, item), list):
                for sub_layer in getattr(self, item):
                    if isinstance(sub_layer, Layer):
                        sub_layer.get_num_params(num_elm_list)
            elif isinstance(getattr(self, item), dict) and item != "__dict__":
                for sub_layer in getattr(self, item).values():
                    if isinstance(sub_layer, Layer):
                        sub_layer.get_num_params(num_elm_list)
            

class Activation(Layer):
    """
    Used for Activation functions that dont have parameters
    """

    def backward_p2(self, inter=False):
        pass

class Sequential:
    def __init__(self, layers: Sequence[Layer], device:str = "cuda"):
        """
        Args:
            layers (Sequence[Layer]): The layers given in order to be executed sequentially
        """
        self.layers = layers

        if device == "cuda":
            self.device = "cuda"
        else:
            self.device = "cpu"
        
    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)

    def init_params(self):
        for layer in self.layers:
            layer.init_params()
    
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward_p1(self, dL_dout: torch.Tensor):
        for layer in self.layers[::-1]:
            dL_dout = layer.backward_p1(dL_dout)
        return dL_dout
    


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, bias: bool = True, device:str = "cuda", dtype=torch.float32):
        """
        Initialize a Dense Layer

        Args:
            input_size (int): size of input vector
            output_size (int): size of output vector
            bias (bool): whether or not to include bias
            device (str): device to use for computation (cpu or cuda)
        
        Attributes:
            weights (torch.Tensor): weights of the dense layer
            bias (torch.Tensor): bias of the dense layer
            weights_g (torch.Tensor): gradients of the weights of the dense layer
            bias_g (torch.Tensor): gradients of the bias of the dense layer
            inputs (torch.Tensor): inputs of the dense layer
            dL_dout (torch.Tensor): derivative of the loss with respect to the dense layers output
        
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias_ = bias
        self.device = device
        self.dtype = dtype
    
    @expose_params({"weights": "weights_g", "bias": "bias_g"}, ["inputs", "dL_dout"])
    def init_params(self):
        self.weights, self.bias = init_params.dense_params(self.input_size, self.output_size, self.bias_, {"device":self.device, "dtype":self.dtype})
        
        self.weights_g = torch.zeros_like(self.weights, device=self.device)
        self.bias_g = torch.zeros_like(self.bias, device=self.device) if self.bias_ else None

        self.inputs = []
        self.dL_dout = []
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the dense layer

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after applying dense layer
        """
        self.inputs.append(x)
        x = vmap(lambda x_: torch.matmul(x_, self.weights))(x)
        if self.bias is not None:
            out = torch.add(x, self.bias)
        else:
            out = x
        return out
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the dense layers output and returns the derivative of the loss with respect to the dense layers input

        Args:
            dL_dout (torch.Tensor): Derivative of the loss with respect to the dense layers output

        Returns:
            torch.Tensor: Derivative of the loss with respect to the dense layers input
        """
        self.dL_dout.append(dL_dout)
        return vmap(lambda dl_dout: torch.matmul(dl_dout, self.weights.T))(dL_dout)
    
    @cleanup_act("dL_dout", "inputs")
    def backward_p2(self, inter=False):
        """
        Calculates the gradients of the dense layer's parameters
        """
        #n = len(self.inputs) if not inter else 1
        if self.multi_stage and not inter:
            dL_dout = torch.cat(self.dL_dout, dim=0)
            inputs = torch.cat(self.inputs, dim=0)
        else:
            dL_dout = self.dL_dout.pop(0)
            inputs = self.inputs.pop(0)

        if inputs.ndim == 2:
            self.weights_g[:] += torch.sum(
                torch.bmm(inputs.unsqueeze(2), 
                            dL_dout.unsqueeze(1)
                    ),
            dim=0)
        elif inputs.ndim == 3:
            self.weights_g[:] += torch.sum(
                torch.bmm(
                    torch.transpose(inputs, -2, -1),
                    dL_dout
                ),
            dim=tuple(range(dL_dout.ndim)[:-2]))
        
        if self.bias is not None:
            self.bias_g[:] += torch.sum(dL_dout, dim=tuple(range(dL_dout.ndim)[:-1]))
            
            #self.dL_dout = None


class ReLU(Activation):

    @expose_params(acts=["inputs"])
    def __init__(self):
        super().__init__()
        self.mask = []
        @torch.jit.script
        def _fwd_op(x:torch.Tensor):
            mask = (x>0)
            return x.mul_(mask), mask
        self.fwd_op = _fwd_op

    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the relu

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after applying relu
        """
        out, mask = self.fwd_op(x)
        #mask = (x>0)
        self.mask.append(mask)
        return out
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the relus output and returns the derivative of the loss with respect to the relus input

        Args:
            dL_dout (torch.Tensor): Derivative of the loss with respect to the dense layers output

        Returns:
            torch.Tensor: Derivative of the loss with respect to the dense layers input
        """
        return dL_dout.mul_(self.mask.pop(0))

class GeLU(Activation):
    @expose_params(acts=["inputs"])
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.sqrt_pi = torch.pi**0.5

        @torch.jit.script
        def _fwd_op(x):
            return x * 0.5 * (1 + torch.erf(x/((2)**(0.5))))
        
        self.fwd_op = _fwd_op

        @torch.jit.script
        def _bwd_op(dL_dout, x):
            return dL_dout * (0.5 * (1 + torch.erf(x/((2)**(0.5)))) + (x)/torch.pi**0.5 * torch.exp(-torch.pow(x, 2)))

        self.bwd_op = _bwd_op
    
    def forward(self, x):
        self.inputs.append(x)
        return F.gelu(x)
        return self.fwd_op(x)

    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        #TODO clean this up 
        input_at_ps = self.inputs.pop(0)
        return self.bwd_op(dL_dout, input_at_ps)
        return dL_dout * (0.5 * (1 + torch.erf(input_at_ps/((2)**(0.5)))) + (input_at_ps)/self.sqrt_pi * torch.exp(-torch.pow(input_at_ps, 2)))

class Dropout(Layer):
    @expose_params(acts=["p_mask"])
    def __init__(self, p: Optional[float]=0.1):
        """
        Initialize a dropout layer

        Args:
            p (Optional[float]): Probability of neurons being dropped. Defaults to 0.1.
        
        Attributes:
            p (Optional[float]): Probability of neurons being dropped. Defaults to 0.1.
            p_mask (torch.Tensor): Mask for the dropout layer
            training (bool): Turned to False during inference in order to turn off dropout
        """
        super().__init__()
        self.p = p
        self.training = True
        self.p_masks = []
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the dropout layer. 
        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output tensor after applying the dropout
        """
        #torch.manual_seed(0)
        if not self.training or self.p==0:
            self.p_masks.append(torch.ones_like(x, device=x.device))
            return x
        if self.p==1:
            self.p_masks.append(torch.ones_like(x, device=x.device))
            return torch.zeros_like(x)
        with torch.no_grad():
            p_mask = torch.bernoulli(torch.ones_like(x, device=x.device) -self.p)
        self.p_masks.append(p_mask)
        return x*p_mask
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the dropouts output and returns the derivative of the loss with respect to the dropouts input

        Args:
            dL_dout (torch.Tensor): Derivative of the loss with respect to the dropout's output

        Returns:
            torch.Tensor: Derivative of the loss with respect to the dropout's input
        """
        if self.p==0:
            return dL_dout
        if self.p==1:
            return torch.zeros_like(dL_dout)
        return dL_dout*self.p_masks.pop(0)


class Embeddings(Layer):
    def __init__(self, num_embeddings, dim, device, dtype=torch.float32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.device = device
        self.dtype = dtype
    
    @expose_params({"weights": "weights_g"}, ["inputs", "dL_dout"])
    def init_params(self):
        self.weights = init_params.embedding_params(self.num_embeddings, self.dim, {"device": self.device, "dtype":self.dtype})
        self.weights_g = torch.zeros_like(self.weights, device=self.device)
        self.inputs = []
        self.dL_dout = []
    
    def forward(self, x):
        self.inputs.append(x)
        return self.weights[x]
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        # Embedding is the first operation so doesnt need a backward pass
        self.dL_dout.append(dL_dout)
        pass
    
    @cleanup_act("dL_dout", "inputs")
    def backward_p2(self, inter=False):
        n = len(self.dL_dout) if not inter else 1
        for _ in range(n):
            dL_dout = self.dL_dout.pop(0)
            inputs = self.inputs.pop(0)
            self.weights_g[inputs] += dL_dout


class Softmax(Activation):
    @expose_params(acts=["out"])
    def __init__(self, n_dims=4):
        super().__init__()
        self.out = []

        @torch.jit.script
        def _per_input_backpass(dldout, s):
            m = s*dldout
            return m - s*torch.sum(m, dim=-1, keepdim=True)
            return s * (dldout - torch.sum(s*dldout, dim=-1, keepdim=True))
        
        self.back_fn = _per_input_backpass
        #for _ in range(n_dims-1):
            #self.back_fn = vmap(self.back_fn)

    def forward(self, x):
        out = torch.softmax(x, dim=-1)
        self.out.append(out)
        return out
    
    
    
    def backward_p1(self, dL_dout:torch.Tensor):
        return self.back_fn(dL_dout, self.out.pop(0))
    


class MultiHeadAttention(Layer):
    #TODO self is the first layer in the model you can cache the linear outputs of the first 3 linears
    def __init__(self, emb_dim: int, num_heads: int, p: Optional[float]=0.1, device="cuda", dtype=torch.float32):
        """
        Initializes the multihead attention

        Args:
            emb_dim (int): Size of the embedding dimension
            num_heads (int): Number of attention heads
            p (Optional[float]): Probability for the dropouts. Defaults to 0.1.
            device (Optional[str]): The device the multi head attention is on. Defaults to "cuda".
        
        
        Attributes:
            emb_dim (int): Size of the embedding dimension
            num_heads (int): Number of attention heads
            linears (Dict[Dense]): The linear layers in the multi head attention
            dropouts (Dict[Dropout]): The dropouts in the multi head attention
            inputs (Dict[torch.Tensor]): The inputs to the multi head attention
            lQ (torch.Tensor): The linear outputs of the Q linear layer
            lK (torch.Tensor): The linear outputs of the K linear layer
            lV (torch.Tensor): The linear outputs of the V linear layer
            sQK_T (torch.Tensor): The softmax of the QK_T matrix
            device (str): The device the multi head attention is on
            streams (List[torch.cuda.Stream]): The streams used to run backward_p2 on the linear layers in parallel
            dL_dout (torch.Tensor): The derivative of the loss with respect to the multi head attention's output
            weights_g (torch.Tensor): The gradient of the weights of the multi head attention
            bias_g (torch.Tensor): The gradient of the bias of the multi head attention
        """
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.inv_sqrt_d = (emb_dim / num_heads)**(-1/2)
        self.p = p

        self.softmax_func = Softmax()
        
        self.device = device
        self.dtype = dtype  

    @expose_params(acts=["lQ", "lK", "lV", "sQK_T"])
    def init_params(self):
        self.linears = {"Q": Dense(self.emb_dim, self.emb_dim, device=self.device, dtype=self.dtype),
                        "K": Dense(self.emb_dim, self.emb_dim, device=self.device, dtype=self.dtype),
                        "V": Dense(self.emb_dim, self.emb_dim, device=self.device, dtype=self.dtype),
                        "O": Dense(self.emb_dim, self.emb_dim, device=self.device, dtype=self.dtype)
                        }
        for l in self.linears.values():
            l.init_params()
        self.dropouts = {"Q": Dropout(p=self.p),
                         "K": Dropout(p=self.p),
                         "V": Dropout(p=self.p)
                        }
        
        self.lQ = []
        self.lK = []
        self.lV = []
        self.sQK_T = []
    
    def forward(self, Q, K, V, mask = None):
        """
        Performs a forward pass through the multihead attention

        Args:
            Q (torch.Tensor): Query tensor
            K (torch.Tensor): Key tensor
            V (torch.Tensor): Value tensor
            mask (Optional[torch.Tensor]): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of multihead attention
        """
        
        lQ:torch.Tensor = self.dropouts["Q"](self.linears["Q"](Q))
        lK = self.dropouts["K"](self.linears["K"](K))
        lV = self.dropouts["V"](self.linears["V"](V))
        
        b, s, d = lQ.shape

        #lQ = torch.cat(lQ.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        #lK = torch.cat(lK.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        #lV = torch.cat(lV.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        lQ = lQ.view(b, s, self.num_heads, -1) 
        lK = lK.view(b, s, self.num_heads, -1)
        lV = lV.view(b, s, self.num_heads, -1)

        self.lQ.append(lQ)
        self.lK.append(lK)
        self.lV.append(lV)
        
        QK_T = vmap(lambda q, k: torch.bmm(q, torch.transpose(k, -1, -2)), in_dims=-2, out_dims=-2)(lQ, lK) * self.inv_sqrt_d
        if mask is not None:
            QK_T = vmap(lambda x: x + mask, in_dims=-2, out_dims=-2)(QK_T)
        
        sQK_T = self.softmax_func(QK_T)
        self.sQK_T.append(sQK_T)
        #torch.softmax(QK_T, dim=-1, out=self.sQK_T) #TODO add inplace update
        
        out = vmap(lambda qk_t, v: torch.bmm(qk_t, v), in_dims=-2, out_dims=-2)(sQK_T, lV)
        out = torch.flatten(out, -2, -1)
        return self.linears["O"](out)

    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the multihead attention output and returns the derivative of the loss with respect to the multihead attention input

        Args:
            dL_dout (torch.Tensor): The derivative of the loss with respect to the multihead attention output

        Returns:
            torch.Tensor: The derivatives of the loss with respect to the multihead attention inputs
        """
        dL_dAtt = self.linears["O"].backward_p1(dL_dout)
        
        dL_dAtt = torch.cat(dL_dAtt.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
        dL_dsQKT = vmap(lambda dl_dout, v: torch.bmm(dl_dout, torch.transpose(v, -1,-2)), in_dims=-2, out_dims=-2)(dL_dAtt, self.lV.pop(0))
        #self.lV = None
        
        # vmap across 3 dims BxCxH 
        dL_dQKT = self.softmax_func.backward_p1(dL_dsQKT)* self.inv_sqrt_d
        
        # TODO verifiy this section
        dL_dlQ = vmap(lambda dl_dqkt, k: torch.bmm(dl_dqkt, k), in_dims=-2, out_dims=-2)(dL_dQKT, self.lK.pop(0))  # k.T not necessary as its k.T.T
        #self.lK = None
        dL_dlKT = vmap(lambda dl_dqkt, q: torch.bmm(torch.transpose(q, -1,-2), dl_dqkt), in_dims=-2, out_dims=-2)(dL_dQKT, self.lQ.pop(0))  
        #self.lQ = None
        dL_dlV = vmap(lambda dl_datt, sqkt: torch.bmm(torch.transpose(sqkt, -2, -1), dl_datt), in_dims=-2, out_dims=-2)(dL_dAtt, self.sQK_T.pop(0)) 
        #self.sQK_T = None
        
        dL_dQ = self.linears["Q"].backward_p1(self.dropouts["Q"].backward_p1(torch.flatten(dL_dlQ, -2, -1)))
        dL_dK = self.linears["K"].backward_p1(self.dropouts["K"].backward_p1(torch.flatten(torch.vmap(lambda dl_dlkt: torch.transpose(dl_dlkt, -1, -2), in_dims=-2, out_dims=-2)(dL_dlKT), -2, -1)))
        dL_dV = self.linears["V"].backward_p1(self.dropouts["V"].backward_p1(torch.flatten(dL_dlV, -2, -1)))
        
        return dL_dQ, dL_dK, dL_dV
             


#TODO will have to update for 3D parallelism (DP)
class BatchNorm2D(Layer):
    def __init__(self, in_channels: int, eps: float=1e-05, momentum =0.1, device:str = "cuda", dtype=torch.float32):
        """
        Initializes a BatchNorm2D layer

        Args:
            in_channels (int): the number of input channels of the forward input 
            eps (float, optional): eps used to prevent div by 0. Defaults to 1e-05.
            momentum (float, optional): the value used for the running_mean and running_var computation. Defaults to 0.1.
            device (str, optional): the device to use for the layer. Defaults to "cuda".

        Attributes:
            in_channels (int): the number of input channels of the forward input 
            eps (float, optional): eps used to prevent div by 0. Defaults to 1e-05.
            momentum (float, optional): the value used for the running_mean and running_var computation. Defaults to 0.1.
            gamma (torch.Tensor): the scaling factor of the input
            beta (torch.Tensor): the bias of the input
            gamma_g (torch.Tensor): the gradient of the scaling factor of the input
            beta_g (torch.Tensor): the gradient of the bias of the input
            r_mean (torch.Tensor): the running mean of the input
            r_var (torch.Tensor): the running variance of the input
            training (bool): whether the layer is in training mode or not
            x_sub_mean (torch.Tensor): the input minus the running mean
            var (torch.Tensor): the variance of the input
            norm_x (torch.Tensor): the normalized input
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.in_channels = in_channels
        self.device = device
        self.dtype = dtype
        self.training = True
        
        @torch.jit.script
        def _fwd_op(x, gamma, beta, eps:float):
            mean, inv_std = torch.batch_norm_stats(x, eps)
            x_s_m = x-mean
            norm_x = x_s_m*inv_std
            return gamma.view(1, -1, 1, 1)*norm_x + beta.view(1, -1, 1, 1)
        
        self.fwd_op = _fwd_op

        @torch.jit.script
        def _bwd_op(dL_dout, gamma, x_s_m, std_inv, B_inv:float):
            dL_dxnorm = dL_dout * gamma.view(1, -1, 1, 1)
            dstd_dxnorm = dL_dxnorm * std_inv
            dL_dvar = (-0.5 * dL_dxnorm * (x_s_m)).sum((0, 2, 3), keepdim=True)  * ((std_inv) ** 3)
            dL_dmean = (-dstd_dxnorm).sum((0, 2, 3), keepdim=True) + (dL_dvar * (-2.0 * (x_s_m)).sum((0, 2, 3), keepdim=True) *B_inv)
            return (dstd_dxnorm) + (2.0 * x_s_m.mul_(dL_dvar).mul_(B_inv)) + (dL_dmean.mul_(B_inv))

        self.bwd_op = _bwd_op


    
    @expose_params({"gamma": "gamma_g", "beta": "beta_g"}, ["norm_x", "dL_dout"])
    def init_params(self):
        self.gamma = torch.ones(self.in_channels, device=self.device, dtype=self.dtype)
        self.beta = torch.zeros(self.in_channels, device=self.device, dtype=self.dtype)
        
        self.gamma_g = torch.zeros_like(self.gamma)
        self.beta_g = torch.zeros_like(self.beta)
        
        self.r_mean = torch.zeros(1, self.in_channels, 1, 1, device=self.device)
        self.r_var = torch.ones(1, self.in_channels, 1, 1, device=self.device)

        self.norm_x = []
        self.dL_dout = []
        
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the batchNorm2D

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: The output of the BAtchNorm2D Layer
        """
        #out = self.fwd_op(x, self.gamma, self.beta, self.eps)
        with torch.enable_grad():
            # NOTE This is the fastest way to do it
            x.requires_grad = True
            out, _, _, _ = torch.cudnn_batch_norm(x, self.gamma, self.beta, None, None, True, 0, epsilon=self.eps)
            self.back_op = out.grad_fn
            out = out.detach()
        self.norm_x.append(out)
        return out
        if self.training:
            mean = x.mean(dim=[0,2,3], keepdim=True)
            var = ((x-mean)**2).mean(dim=[0,2,3], keepdim=True)
            #self.r_mean = (1 - self.momentum) * self.r_mean + self.momentum * mean
            #self.r_var = (1 - self.momentum) * self.r_var + self.momentum * var
        else:
            mean, var = self.r_mean, self.r_var
        
        x_sub_mean = x - mean
        self.x_sub_mean.append(x_sub_mean)
        #var = var + self.eps
        norm_x = F.batch_norm(x) #(x_sub_mean)/torch.sqrt(var)
        out = self.gamma.view(1, self.in_channels, 1, 1)*norm_x + self.beta.view(1, self.in_channels, 1, 1)
        return F.batch_norm(x)
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the BatchNorm2D output and returns the derivative of the loss with respect to the BatchNorm2D input

        Args:
            dL_dout (torch.Tensor): The derivative of the loss with respect to the BatchNorm2D output

        Returns:
            torch.Tensor: The derivatives of the loss with respect to the BatchNorm2D inputs
        """
        
        self.dL_dout.append(dL_dout)
        return self.back_op(dL_dout)[0]

        # https://stackoverflow.com/questions/67968913/derivative-of-batchnorm2d-in-pytorch
        in_shape = dL_dout.shape
        B = in_shape[0]*in_shape[2]*in_shape[3]

        #x_s_m = self.x_sub_mean.pop(0)
        #std_inv = self.std_inv.pop(0)
        
        return self.bwd_op(dL_dout, self.gamma, x_s_m, std_inv, 1/B)
        
        dL_dxnorm = dL_dout * self.gamma.view(1, -1, 1, 1)
        
        v = (x_s_m**2).mean(dim=[0,2,3], keepdim=True) + self.eps
        self.norm_x.append((x_s_m)/torch.sqrt(v))
        dL_dvar = (-0.5 * dL_dxnorm * (x_s_m)).sum((0, 2, 3), keepdim=True)  * ((v) ** -1.5)
        dL_dmean = (-1.0 / torch.sqrt(v) * dL_dxnorm).sum((0, 2, 3), keepdim=True) + (dL_dvar * (-2.0 * (x_s_m)).sum((0, 2, 3), keepdim=True) / B)
        return (dL_dxnorm / torch.sqrt(v)) + (2.0 * dL_dvar * (x_s_m) / B) + (dL_dmean / B)
    
        
    @cleanup_act("dL_dout", "norm_x")
    def backward_p2(self, inter=False):
        """
        Calculates the gradients of the parameters in the BatchNorm2D
        """
        #n = len(self.dL_dout) if not inter else 1
        if self.multi_stage and not inter:
            dL_dout = torch.cat(self.dL_dout, dim=0)
            norm_x = torch.cat(self.norm_x, dim=0)
        else:
            dL_dout = self.dL_dout.pop(0)
            norm_x = self.norm_x.pop(0)

        self.gamma_g[:] += torch.sum(dL_dout*norm_x, dim=[0,2,3])
        self.beta_g[:] += torch.sum(dL_dout, dim=[0,2,3])

      
class NLPLayerNorm(Layer):
    def __init__(self, dim: int, dim_size: int, eps:Optional[float]=1e-08, device: Optional[str]="cuda", dtype=torch.float32):
        """
        Initializes the NLP Layer Norm

        Args:
            dim (int): Embedding dim
            dim_size (int): Size of embedding dim
            eps (Optional[float]): eps used to prevent div by 0. Defaults to 1e-08.
            device (Optional[str]): the device to use for the layer. Defaults to "cuda".

        Attributes:
            dim (int): Embedding dim
            dim_size (int): Size of embedding dim
            eps (Optional[float]): eps used to prevent div by 0. Defaults to 1e-08.
            gamma (torch.Tensor): gamma parameter
            bias (torch.Tensor): bias parameter
            gamma_g (torch.Tensor): gamma gradient
            bias_g (torch.Tensor): bias gradient
            x_sub_mean (torch.Tensor): x - u
            var (torch.Tensor): variance + eps
            norm_x (torch.Tensor): normalization of x 
            dL_dout (torch.Tensor): the derivative of the loss with respect to the output
        """
        super().__init__()
        self.dim = dim #seqdim for nlp
        self.dim_size = dim_size
        self.eps = eps
        self.device = device
        self.dtype = dtype
    
    @expose_params({"gamma": "gamma_g", "beta": "beta_g"}, ["x_sub_mean", "var", "norm_x", "dL_dout"])
    def init_params(self):
        self.gamma = torch.ones(self.dim_size, device=self.device, dtype=self.dtype)
        self.bias = torch.zeros(self.dim_size, device=self.device, dtype=self.dtype)
        
        self.gamma_g = torch.zeros_like(self.gamma)
        self.bias_g = torch.zeros_like(self.bias)

        self.x_sub_mean = []
        self.var = []
        self.norm_x = []
        self.dL_dout = []
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the NLP Layer Norm

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: The output of the NLP layer norm
        """

        mean = torch.mean(x, dim=self.dim, keepdim=True)
        x_sub_mean = x-mean
        self.x_sub_mean.append(x_sub_mean)
        var = torch.mean(torch.square(x_sub_mean), dim=self.dim, keepdim=True) + self.eps
        self.var.append(var)
        norm_x = (x_sub_mean)/torch.sqrt(var)
        self.norm_x.append(norm_x)
        return norm_x*self.gamma.view(1,1,-1) + self.bias.view(1,1,-1)
    

    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the NLP layer norm output and returns the derivative of the loss with respect to the NLP layer norm input

        Args:
            dL_dout (torch.Tensor): The derivative of the loss with respect to the NLP layer norm output

        Returns:
            torch.Tensor: The derivatives of the loss with respect to the NLP layer norm inputs
        """
        self.dL_dout.append(dL_dout)
        x_s_m = self.x_sub_mean.pop(0)
        v = self.var.pop(0)
        
        dx_hat = dL_dout * self.gamma.view(1,1,-1)  
        dL_dvar = torch.sum(dx_hat * x_s_m, dim=-1, keepdim=True) * -.5 * v**(-1.5)
        dL_dmean = torch.sum(dx_hat/-torch.sqrt(v), dim=-1, keepdim=True) + dL_dvar * torch.mean(-2. * x_s_m, dim=-1, keepdim=True)
        return dx_hat / torch.sqrt(v) + dL_dvar * 2. * x_s_m /self.dim_size + dL_dmean / self.dim_size
        

    @cleanup_act("dL_dout", "norm_x")
    def backward_p2(self, inter=False):
        """
        Computes the gradients of the parameter in the NLP layer norm
        """
        n = len(self.dL_dout) if not inter else 1
        for _ in range(n):
            dL_dout = self.dL_dout.pop(0)
            norm_x = self.norm_x.pop(0)
            self.bias_g[:] += torch.sum(dL_dout, dim=tuple(range(dL_dout.ndim)[:-1]))
            self.gamma_g[:] += torch.sum(dL_dout*norm_x, dim=tuple(range(dL_dout.ndim)[:-1]))
      
        
class NLPRMSNorm(Layer):
    def __init__(self, dim: int, dim_size: int, eps: float= 1e-08, device: Optional[str]= "cuda", dtype=torch.float32):
        """
        Initializes the NLP RMS Norm

        Args:
            dim (int): Embedding dim
            dim_size (int): Size of embedding dim
            eps (Optional[float]): eps used to prevent div by 0. Defaults to 1e-08.
            device (Optional[str]): the device to use for the layer. Defaults to "cuda".
            

        Attributes:
            dim (int): Embedding dim
            dim_size (int): Size of embedding dim
            eps (Optional[float]): eps used to prevent div by 0. Defaults to 1e-08.
            weights (torch.Tensor): weights parameter
            weights_g (torch.Tensor): weights gradient
            inputs (torch.Tensor): input tensor
            mean_pow2 (torch.Tensor): mean of x^2
            rms_norm_x (torch.Tensor): rms normalized x
            dL_dout (torch.Tensor): the derivative of the loss with respect to the output
        """
        super().__init__()
        self.dim = dim #seqdim for nlp
        self.dim_size = dim_size
        self.eps = eps
        self.device = device
        self.dtype = dtype
        
        @torch.jit.script
        def _per_input_backpass(dldout, irms, z):
            n = dldout.size(-1)
            return irms*((-z/n) * torch.sum(dldout*z, dim=-1, keepdim=True) + dldout)
    
        self.back_fn = _per_input_backpass
    
    @expose_params({"weights": "weights_g"}, ["IRMS",  "rms_norm_x", "rms_norm_x_p2", "dL_dout"])
    def init_params(self):
        self.weights = torch.ones(self.dim_size, device=self.device, dtype=self.dtype)
        self.weights_g = torch.zeros_like(self.weights)

        self.IRMS = []
        self.rms_norm_x = []
        self.rms_norm_x_p2 = []
        self.dL_dout = []

    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the NLP RMS norm

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: The output of the NLP RMS norm
        """
        u_x2 = torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        IRMS = torch.rsqrt(u_x2)  # Inverse Root Mean Square
        self.IRMS.append(IRMS)
        rms_norm_x = x*IRMS
        self.rms_norm_x.append(rms_norm_x)
        return rms_norm_x*self.weights

    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the NLP RMS norm output and returns the derivative of the loss with respect to the NLP RMS norm input

        Args:
            dL_dout (torch.Tensor): The derivative of the loss with respect to the NLP RMS norm output

        Returns:
            torch.Tensor: The derivatives of the loss with respect to the NLP RMS norm inputs
        """

        self.dL_dout.append(dL_dout)
        rms_norm_x = self.rms_norm_x.pop(0)
        self.rms_norm_x_p2.append(rms_norm_x)
        return self.back_fn(dL_dout, self.IRMS.pop(0), rms_norm_x)
    
    @cleanup_act("dL_dout", "rms_norm_x_p2")
    def backward_p2(self, inter=False):
        """
        Computes the gradients of the parameter in the NLP layer norm
        """
        #for _ in range(len(self.dL_dout)):
        if self.multi_stage and not inter:
            dL_dout = torch.cat(self.dL_dout)
            rms_norm_x = torch.cat(self.rms_norm_x_p2)
        else:
            dL_dout = self.dL_dout.pop(0)
            rms_norm_x = self.rms_norm_x_p2.pop(0)
        self.weights_g[:] += torch.sum(dL_dout*rms_norm_x, dim=tuple(range(dL_dout.ndim)[:-1]))


class BertEmbeddings(Layer):
    def __init__(self, dim, vocab_size, seq_len, device="cuda", dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype
    
    def init_params(self):
        self.token_emb = Embeddings(self.vocab_size, self.dim, device=self.device, dtype=self.dtype)
        self.segmnet_emb = Embeddings(2, self.dim, device=self.device, dtype=self.dtype)
        self.pos_emb = torch.zeros(self.seq_len, self.dim, device=self.device, dtype=self.dtype)
        
        for pos in range(self.seq_len):
            for i in range(0, self.dim, 2):
                self.pos_emb[pos, i] = math.sin(pos/ (10000**((2*i)/self.dim)))
                self.pos_emb[pos, i+1] = math.cos(pos/ (10000**((2*(i+1))/self.dim)))
        
        self.pos_emb = self.pos_emb.unsqueeze(0)
        
        self.norm = NLPRMSNorm(-1, self.dim, device=self.device, dtype=self.dtype)
        self.token_emb.init_params()
        self.segmnet_emb.init_params()
        self.norm.init_params()
    
    def forward(self, x, seg_mask):
        x = self.token_emb(x) + self.segmnet_emb(seg_mask) + self.pos_emb
        return self.norm(x)

    def backward_p1(self, dL_dout):
        dL_dout = self.norm.backward_p1(dL_dout)
        self.token_emb.backward_p1(dL_dout)
        self.segmnet_emb.backward_p1(dL_dout)
        return dL_dout
        
        
        
class BertBlock(Layer):
    def __init__(self, emb_dim: int, num_heads: int, dim_ff:int, activation:Layer=ReLU, eps:float=1e-08, p:float=0.1, device = "cuda", dtype=torch.float32):
        """
        Initialize BERT block

        Args:
            emb_dim (int): size of the embedding dim
            num_heads (int): number of heads in the multihead attention
            dim_ff (int): size of hidden dim in the ffn
            activation (Layer, optional): activation function used on hidden layer in ffn. Defaults to ReLU.
            eps (float, optional): the eps used in the layer norms. Defaults to 1e-08.
            p (float, optional): the probability used in the dropouts. Defaults to 0.1.
            device (str optional): the device used for computation. Defaults to "cuda".

        Attributes:
            multihead (MultiHeadAttention): multihead attention layer
            linears (Dict[Dense]): linear layers
            ff_act (Layer): activation function used on hidden layer in ffn
            norms (Dict[NLPLayerNorm]): layer norms
            dropouts (Dict[Dropout]): dropouts
            device (str): device used for computation
            streams (List[torch.cuda.Stream]): streams used for parallel computation of backward_p2
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dim_ff = dim_ff
        self.activation = activation
        self.eps = eps
        self.p = p
        self.device = device
        self.dtype = dtype
    
    def init_params(self):
        self.multihead = MultiHeadAttention(emb_dim=self.emb_dim, num_heads=self.num_heads, device=self.device, dtype=self.dtype)
        self.multihead.init_params()
        self.linears = {0: Dense(self.emb_dim, self.dim_ff, device=self.device, dtype=self.dtype),
                        1: Dense(self.dim_ff, self.emb_dim, device=self.device, dtype=self.dtype)}
        for l in self.linears.values():
            l.init_params()
        self.ff_act: Activation = self.activation()
        self.norms = {"multi_head": NLPRMSNorm(-1, self.emb_dim, eps=self.eps, device=self.device, dtype=self.dtype),
                      "ff": NLPRMSNorm(-1, self.emb_dim, eps=self.eps, device=self.device, dtype=self.dtype)}
        for n in self.norms.values():
            n.init_params()
        self.dropouts = {"multi_head": Dropout(p=self.p),
                         "ff": Dropout(p=self.p)}

    def forward(self, x:torch.Tensor):
        """
        Performs a forward pass through the BERT block. 

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor of Bert block 
        """
        mh_out = self.multihead(x, x, x)
        mh_out = self.dropouts["multi_head"](mh_out)
        norm_mh_out = self.norms["multi_head"](mh_out + x)
        
        ff1 = self.linears[0](norm_mh_out)
        a = self.ff_act(ff1)
        ff2 = self.linears[1](a)
        ff2 = self.dropouts["ff"](ff2)
        return self.norms["ff"](ff2 + norm_mh_out)
    

    @multi_stage_wrapper
    def backward_p1(self, dL_dout:torch.Tensor):
        """
        Takes the derivative of the loss with respect to the BERT block output and returns the derivative of the loss with respect to the BERT block input

        Args:
            dL_dout (torch.Tensor): Takes the derivative of the loss with respect to the BERT block output 

        Returns:
            torch.Tensor: The derivative of the loss with respect to the BERT block input

        """
        dL_dff2_d = self.norms["ff"].backward_p1(dL_dout)
        dL_dff2 = self.dropouts["ff"].backward_p1(dL_dff2_d)
        dL_da = self.linears[1].backward_p1(dL_dff2)
        dL_dff1 = self.ff_act.backward_p1(dL_da)
        dL_dnormmhout = self.linears[0].backward_p1(dL_dff1) + dL_dff2_d

        dL_dmhout_d = self.norms["multi_head"].backward_p1(dL_dnormmhout)
        dL_dmhout = self.dropouts["multi_head"].backward_p1(dL_dmhout_d)
        dL_din = torch.sum(torch.stack(self.multihead.backward_p1(dL_dmhout)),dim=0)
        return dL_din + dL_dmhout_d

class BertLoss(Layer):
    def __init__(self):
        super().__init__()
        self.ns_loss = loss.NLPCrossEntropyLoss()
        self.mask_loss = loss.NLPCrossEntropyLoss()

    def forward(self, x, y):
        ns_x, mask_x = x
        ns_y, mask_y = y
        return self.ns_loss(ns_x, ns_y) + self.mask_loss(mask_x, mask_y)

    def backward_p1(self, dL_dout):
        return self.ns_loss.backward(), self.mask_loss.backward()
        

class BertOutputLoss(Layer):
    def __init__(self, dim, vocab_size, device = "cuda", dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.mask_linear = Dense(dim, vocab_size, device=device, dtype=dtype)
        self.ns_linear = Dense(dim, 2, device=device, dtype=dtype)
        self.loss_fn = BertLoss()
        self.device = device
        self.dtype = dtype
    
    @expose_params(acts=["mask"])
    def init_params(self):
        self.mask_linear.init_params()
        self.ns_linear.init_params()
        self.mask = []

    def forward(self, x, y, mask):
        self.in_shape = x.shape
        self.mask.append(mask)
        ns_x = self.ns_linear(x[:,0].unsqueeze(1))
        mask_x = self.mask_linear(x[:, mask])
        return self.loss_fn((ns_x, mask_x), y)
    
    def backward_p1(self, dL_dout=None):
        dL_dnsout, dL_dmaskout = self.loss_fn.backward_p1(dL_dout)
        dL_din = torch.zeros(self.in_shape, device = self.device, dtype=self.dtype)
        dL_din[:, 0] = self.ns_linear.backward_p1(dL_dnsout).squeeze(1)
        dL_din[:, self.mask.pop(0)] = self.mask_linear.backward_p1(dL_dmaskout)
        return dL_din
    


class Flatten(Activation):
    def forward(self, x: torch.Tensor):
        self.shape = x.shape
        return x.view(x.size(0), -1)

    def backward_p1(self, dL_dout):
        return dL_dout.view(*self.shape)

def complex_matmul(a: torch.Tensor, b: torch.Tensor, groups: int = 1) -> torch.Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    #real = a.real @ b.real - a.imag @ b.imag
    #imag = a.imag @ b.real + a.real @ b.imag
    c = torch.matmul(a, b)
    c = torch.movedim(c, c.dim() - 1, 2).squeeze(-1)
    #real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    #imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    #c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    #c.real, c.imag = real, imag
    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Sequence[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Sequence):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor = None,
    padding: Union[int, Sequence[int], str] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Sequence[int]] = 1,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int], str) If int, Number of zero samples to pad then
            input on the last dimension. If str, "same" supported to pad input for size preservation.
        padding_mode: (str) Padding mode to use from {constant, reflection, replication}.
                      reflection not available for 3d.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
        dilation: (Union[int, Iterable[int]) Dilation rate for the kernel.
        groups: (int) Number of groups for the convolution.

    Returns:
        (Tensor) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    signal = signal.unsqueeze(0)
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(padding, str):
        if padding == "same":
            if stride != 1 or dilation != 1:
                raise ValueError("stride must be 1 for padding='same'.")
            padding_ = [(k - 1) / 2 for k in kernel.shape[2:]]
        else:
            raise ValueError(f"Padding mode {padding} not supported.")
    else:
        padding_ = to_ntuple(padding, n=n)

    # internal dilation offsets
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors (round to support even sized convolutions)
    signal_padding = [r(p) for p in padding_[::-1] for r in (math.floor, math.ceil)]
    signal = F.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    signal_size = signal.size()  # original signal size without padding to even
    if signal.size(-1) % 2 != 0:
        signal = F.pad(signal, [0, 1])

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.size(i) - kernel.size(i)]
    ]
    padded_kernel = F.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = torch.fft.rfftn(signal.float(), dim=tuple(range(2, signal.ndim)))
    kernel_fr = torch.fft.rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = torch.fft.irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    crop_slices = [slice(None), slice(None)] + [
        slice(0, (signal_size[i] - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output.squeeze(0)

class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, k_size: Union[Sequence[int], int], bias: bool=True,  padding: Union[bool, int]=False, stride: Union[Sequence[int], int]=1, device: Optional[str]="cuda", dtype=torch.float32):
        """
        Initializes Conv2D layer

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            k_size (Union[Sequence[int], int]): size of kernel
            bias (bool): whether the conv2d layer has bias. Defaults to True.
            padding (Union[bool, int]): the padding either side of the input feature. Defaults to False.
            stride (Union[Sequence[int], int], optional): the stride pattern followed by the kernel. Defaults to 1.
            device (Optioanl[str]): the device used for computation. Defaults to "cuda".
            

        Raises:
            NotImplementedError: There is a Bug in the backward pass when the stride is greater than 3
            
        Attributes:
            kernel (torch.Tensor): the kernel of the conv2d layer
            bias (torch.Tensor): the bias of the conv2d layer
            kernel_g (torch.Tensor): the gradient of the kernel of the conv2d layer
            bias_g (torch.Tensor): the gradient of the bias of the conv2d layer
            padding (int): the padding either side of the input feature
            stride (Tuple[int]): the stride pattern followed by the kernel
            inputs (torch.Tensor): the input tensor
        """
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.bias_ = bias

        if isinstance(k_size, int):
            self.k_size = (k_size, k_size)
        
        self.device = device
        self.dtype = dtype
        
        if isinstance(padding, bool):
            self.padding = 1 if padding else 0
        else:
            self.padding = padding

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        if self.stride[0] > 3 or self.stride[1] > 3:
            raise NotImplementedError("Stride larger than 3 not implemented")
    

    @expose_params({"kernel":"kernel_g", "bias":"bias_g"}, ["inputs", "dL_dout"])
    def init_params(self):
        self.kernel, self.bias = init_params.conv2d_params(self.in_channels, self.out_channels, self.k_size, bias=self.bias_, factory_kwargs={"device": self.device, "dtype":self.dtype})

        self.kernel_g = torch.zeros_like(self.kernel)
        if self.bias_:
            self.bias_g = torch.zeros_like(self.bias)
        
        self.inputs = []
        self.dL_dout = []


    def forward(self, x:torch.Tensor):
        """
        Preforms a forward pass through the Conv2D layer

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor
        """
        self.inputs.append(x)
        return F.conv2d(x, self.kernel, self.bias, stride=self.stride, padding=self.padding)
        print(dist.get_rank(), out.shape)
        return out
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the Conv2D Layer's output and returns the derivative of the loss with respect to the Conv2D Layer's input

        Args:
            dL_dout (torch.Tensor): derivative of the loss with respect to the Conv2D Layer's output

        Returns:
            torch.Tensor: derivative of the loss with respect to the Conv2D Layer's input
        """
        self.dL_dout.append(dL_dout)
        #dL_din = F.grad.conv2d_input(self.inputs[0].shape, self.kernel, dL_dout, stride=self.stride, padding=self.padding)
        dL_din = F.conv_transpose2d(dL_dout, self.kernel, stride=self.stride, padding=self.padding, output_padding=(self.stride[0]-1,self.stride[1]-1))
        return dL_din
    
    @cleanup_act("dL_dout", "inputs")
    def backward_p2(self, inter=False):
        """
        Computes the gradients of the parameter within the Conv2D layer
        """
        #n = len(self.dL_dout) if not inter else 1
        if self.multi_stage and not inter:
            dL_dout = torch.cat(self.dL_dout, dim=0)
            inputs = torch.cat(self.inputs, dim=0)
        else:
            dL_dout = self.dL_dout.pop(0)
            inputs = self.inputs.pop(0)

        if isinstance(self.bias, torch.Tensor):
            self.bias_g[:] += torch.sum(dL_dout, dim=(0, 2, 3))
        
        
        self.kernel_g[:] += F.grad.conv2d_weight(inputs, self.kernel_g.shape, dL_dout, stride=self.stride, padding=self.padding)


class Conv2DTranspose(Layer):
    ...
    #Have Fun ;)


class MaxPool2D(Layer):
    @expose_params(acts=["indices"])
    def __init__(self, k_size: Union[Sequence[int], int], padding: Union[bool, int]=False, stride: Union[Sequence[int], int]=1):
        """
        Initializes a MaxPool2D layer

        Args:
            k_size (Union[Sequence[int], int]): Size of pool kernel
            padding (Union[bool, int]): the padding either side of the input feature. Defaults to False.
            stride (Union[Sequence[int], int], optional): the stride pattern followed by the kernel. Defaults to 1.
        
        Attributes:
            k_size (Tuple[int]): Size of pool kernel)
            padding (int): the padding either side of the input feature
            stride (Tuple[int]): the stride pattern followed by the kernel
            indices (torch.Tensor): the indices of the max pool
        """
        super().__init__()
        if isinstance(k_size, int):
            self.k_size = (k_size, k_size)
        
        if isinstance(padding, bool):
            self.padding = 1 if padding else 0
        else:
            self.padding = padding

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        self.indices = []
    
    def forward(self, x: torch.Tensor):
        """
        Preforms a forward pass through the MaxPool2D layer

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        self.in_shape = x.shape
        out, indices = F.max_pool2d(x, kernel_size=self.k_size, stride=self.stride, padding=self.padding, return_indices=True)
        self.indices.append(indices)
        return out
    
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the MaxPool2D Layer's output and returns the derivative of the loss with respect to the MaxPool2D Layer's input

        Args:
            dL_dout (torch.Tensor): derivative of the loss with respect to the MaxPool2D Layer's output

        Returns:
            torch.Tensor: derivative of the loss with respect to the MaxPool2D Layer's input
        """
        out = F.max_unpool2d(dL_dout, self.indices.pop(0), kernel_size=self.k_size, stride=self.stride, padding=self.padding, output_size=self.in_shape)
        return out



class AvgPool2D(Layer):
    def __init__(self, k_size: Union[Sequence[int], int], padding: Union[bool, int]=False, stride: Union[Sequence[int], int]=None):
        """
        Initializes a AvgPool2D layer

        Args:
            k_size (Union[Sequence[int], int]): Size of pool kernel
            padding (Union[bool, int]): the padding either side of the input feature. Defaults to False.
            stride (Union[Sequence[int], int], optional): the stride pattern followed by the kernel. Defaults to 1.
        
        Attributes:
            k_size (Tuple[int]): Size of pool kernel)
            padding (int): the padding either side of the input feature
            stride (Tuple[int]): the stride pattern followed by the kernel
            indices (torch.Tensor): the indices of the max pool
        """
        super().__init__()
        if isinstance(k_size, int):
            self.k_size = (k_size, k_size)
        
        if isinstance(padding, bool):
            self.padding = 1 if padding else 0
        else:
            self.padding = padding
        
        if stride is None:
            self.stride = self.k_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
        if self.stride[0] != self.k_size[0]:
            raise NotImplementedError("A stride size not equal to the kernel size has not been implmented due to the interpolation used in the backward pass")
    
    def forward(self, x: torch.Tensor):
        """
        Preforms a forward pass through the AvgPool2D layer

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return F.avg_pool2d(x, kernel_size=self.k_size, stride=self.stride, padding=self.padding)
    
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the AvgPool2D Layer's output and returns the derivative of the loss with respect to the AvgPool2D Layer's input

        Args:
            dL_dout (torch.Tensor): derivative of the loss with respect to the AvgPool2D Layer's output

        Returns:
            torch.Tensor: derivative of the loss with respect to the AvgPool2D Layer's input
        """
        intrp_size = (dL_dout.shape[2]*self.k_size[0], dL_dout.shape[3]*self.k_size[1])
        dL_dout = F.interpolate(dL_dout, size=intrp_size)
        return dL_dout * (1/(self.k_size[0]*self.k_size[1]))
    


class BasicResNetBlock(Layer):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, device: str="cuda", dtype=torch.float32):
        """
        Initializes a Basic ResNet Block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): he stride pattern followed by the kernel of the first conv layer. Defaults to 1.
            device (Optional, str): Device used to parallelize the execution of the Basic ResNet Block. Defaults to "cuda".
        
        Attributes:
            convs (List[Conv2D]): List of Conv2D layers
            batchnorms (List[BatchNorm2D]): List of BatchNorm2D layers)
            relus (List[ReLU]): List of ReLU layers])
            shortcut (List[Layer]): List of layers that are used to connect the input and output of the Basic ResNet Block
            streams (List[torch.cuda.Stream]): List of streams used to parallelize the execution of the Basic ResNet Block
            device (str): Device used to parallelize the execution of the Basic ResNet Block
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.device = device
        self.dtype = dtype
    
    def init_params(self):
        self.convs = [Conv2D(self.in_channels, self.out_channels, 3, stride=self.stride, padding=1, bias=False, device=self.device, dtype=self.dtype),
                      Conv2D(self.out_channels, self.out_channels, 3, stride=1, padding=1, bias=False, device=self.device, dtype=self.dtype)
                      ]
        for c in self.convs:
            c.init_params()
        self.batchnorms = [BatchNorm2D(self.out_channels, device=self.device, dtype=self.dtype),
                           BatchNorm2D(self.out_channels, device=self.device, dtype=self.dtype)
                           ]
        for b in self.batchnorms:
            b.init_params()
        self.relus = [ReLU(), ReLU()]
        if self.stride != 1 or self.in_channels != self.out_channels*self.expansion:
            self.shortcut = [Conv2D(self.in_channels, self.out_channels*self.expansion, 1, stride=self.stride, bias=False, device=self.device, dtype=self.dtype),
                             BatchNorm2D(self.out_channels*self.expansion, device=self.device, dtype=self.dtype)
                             ]
            for l in self.shortcut:
                l.init_params()
        else:
            self.shortcut = []
        
    def forward(self, x: torch.Tensor):
        """
        Preforms a forward pass through the BasicResNetBlock

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        out = self.convs[0](x)
        out = self.batchnorms[0](out)
        out = self.relus[0](out)
        out = self.convs[1](out)
        out = self.batchnorms[1](out)
        if self.shortcut:
            for layer in self.shortcut:
                x = layer(x)
        return self.relus[1](out + x)
    

    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the BasicResNetBlock's output and returns the derivative of the loss with respect to the BasicResNetBlock's input

        Args:
            dL_dout (torch.Tensor): derivative of the loss with respect to theBasicResNetBlock's output

        Returns:
            torch.Tensor: derivative of the loss with respect to the BasicResNetBlock's input
        """
        dL_dshortcut = self.relus[1].backward_p1(dL_dout)
        dL_dconv1 = self.batchnorms[1].backward_p1(dL_dshortcut)
        dL_drelu0 = self.convs[1].backward_p1(dL_dconv1)
        dL_dbn0 = self.relus[0].backward_p1(dL_drelu0)
        dL_dconv0 = self.batchnorms[0].backward_p1(dL_dbn0)
        dL_din1 = self.convs[0].backward_p1(dL_dconv0)
        
        dL_din2 = dL_dshortcut
        for layer in self.shortcut[::-1]:
            dL_din2 = layer.backward_p1(dL_din2)
        
        return dL_din1 + dL_din2
        

class ResNetBottleneck(Layer):
    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, device: str="cuda", dtype=torch.float32):
        """
        Initializes a Basic ResNet Block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): he stride pattern followed by the kernel of the first conv layer. Defaults to 1.
            device (str, optional): Device used to parallelize the execution of the Basic ResNet Block. Defaults to "cuda".
        
        Attributes:
            convs (List[Conv2D]): List of Conv2D layers
            batchnorms (List[BatchNorm2D]): List of BatchNorm2D layers)
            relus (List[ReLU]): List of ReLU layers])
            shortcut (List[Layer]): List of layers that are used to connect the input and output of the Basic ResNet Block
            streams (List[torch.cuda.Stream]): List of streams used to parallelize the execution of the Basic ResNet Block
            device (str): Device used to parallelize the execution of the Basic ResNet Block
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.device = device
        self.dtype = dtype
    

    def init_params(self):
        self.convs = [Conv2D(self.in_channels, self.out_channels, 1, bias=False, device=self.device, dtype=self.dtype),
                      Conv2D(self.out_channels, self.out_channels, 3, stride=self.stride, padding=1, bias=False, device=self.device, dtype=self.dtype),
                      Conv2D(self.out_channels, self.expansion*self.out_channels, 1, bias=False, device=self.device, dtype=self.dtype)
                      ]
        for c in self.convs:
            c.init_params()
        self.batchnorms = [BatchNorm2D(self.out_channels, device=self.device, dtype=self.dtype),
                           BatchNorm2D(self.out_channels, device=self.device, dtype=self.dtype),
                           BatchNorm2D(self.expansion*self.out_channels, device=self.device, dtype=self.dtype)
                           ]
        for b in self.batchnorms:
            b.init_params()
        self.relus = [ReLU(), ReLU(), ReLU()]
        if self.stride != 1 or self.in_channels != self.out_channels*self.expansion:
            self.shortcut = [Conv2D(self.in_channels, self.out_channels*self.expansion, 1, stride=self.stride, bias=False, device=self.device, dtype=self.dtype),
                             BatchNorm2D(self.out_channels*self.expansion, device=self.device, dtype=self.dtype)
                             ]
            for l in self.shortcut:
                l.init_params()
        else:
            self.shortcut = []
    
    def forward(self, x: torch.Tensor):
        """
        Preforms a forward pass through the ResNetBottleneck

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        out = self.convs[0](x)
        out = self.batchnorms[0](out)
        out = self.relus[0](out)
        out = self.convs[1](out)
        out = self.batchnorms[1](out)
        out = self.relus[1](out)
        out = self.convs[2](out)
        out = self.batchnorms[2](out)
    
        if self.shortcut:
            for layer in self.shortcut:
                x = layer(x)
        return self.relus[2](out + x)
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        """
        Takes the derivative of the loss with respect to the ResNetBottleneck's output and returns the derivative of the loss with respect to the ResNetBottleneck's input

        Args:
            dL_dout (torch.Tensor): derivative of the loss with respect to the ResNetBottleneck's output

        Returns:
            torch.Tensor: derivative of the loss with respect to the ResNetBottleneck's input
        """
        dL_dshortcut = self.relus[2].backward_p1(dL_dout)
        dL_dconv2 = self.batchnorms[2].backward_p1(dL_dshortcut)
        dL_drelu1 = self.convs[2].backward_p1(dL_dconv2)
        dL_dbn1 = self.relus[1].backward_p1(dL_drelu1)
        dL_dconv1 = self.batchnorms[1].backward_p1(dL_dbn1)
        dL_drelu0 = self.convs[1].backward_p1(dL_dconv1)
        dL_dbn0 = self.relus[0].backward_p1(dL_drelu0)
        dL_dconv0 = self.batchnorms[0].backward_p1(dL_dbn0)
        dL_din1 = self.convs[0].backward_p1(dL_dconv0)
        dL_din2 = dL_dshortcut
        for layer in self.shortcut[::-1]:
            dL_din2 = layer.backward_p1(dL_din2)
        
        return dL_din1 + dL_din2

        
class ResNet(Layer):
    # For Reference https://github.com/henryqin1997/CIFAR10-ResNet50-PyTorch/blob/master/models/resnet.py
    # Needs to be parallised just for testing
    def __init__(self, block: Union[BasicResNetBlock , ResNetBottleneck], num_blocks: list, num_classes: int=10, device:str = "cuda", dtype=torch.float32):
        super().__init__()
        self.in_planes = 64

        self.device = device
    
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.dtype = dtype
    
    def init_params(self):
        self.conv1 = Conv2D(3, 64, 3, stride=1, padding=1, bias=False, device=self.device, dtype=self.dtype)
        self.conv1.init_params()
        self.bn1 = BatchNorm2D(64, device=self.device, dtype=self.dtype)
        self.bn1.init_params()
        
        self.layers1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layers1.init_params()
        self.layers2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layers2.init_params()
        self.layers3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layers3.init_params()
        self.layers4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.layers4.init_params()
        
        self.linear = Dense(512*self.block.expansion, self.num_classes, device=self.device, dtype=self.dtype)
        self.linear.init_params()
        self.flatten = Flatten()
        self.relu = ReLU()
        self.avgpool = AvgPool2D(4)
        
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, device=self.device, dtype=self.dtype))
            self.in_planes = planes * block.expansion
        return Sequential(layers, device=self.device)
    
    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.layers4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        dL_davgpool = self.linear.backward_p1(dL_dout)
        dL_davgpool = self.flatten.backward_p1(dL_davgpool)
        dL_dl4 = self.avgpool.backward_p1(dL_davgpool)
        dL_dl3 = self.layers4.backward_p1(dL_dl4)
        dL_dl2 = self.layers3.backward_p1(dL_dl3)
        dL_dl1 = self.layers2.backward_p1(dL_dl2)
        dL_drelu = self.layers1.backward_p1(dL_dl1)
        dL_dbn = self.relu.backward_p1(dL_drelu)
        dL_dconv = self.bn1.backward_p1(dL_dbn)
        return self.conv1.backward_p1(dL_dconv)


class SiLU(Activation):
    @expose_params(acts=["inputs"])
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x: torch.Tensor):
        self.inputs.append(x)
        return x * torch.sigmoid(x)

    def backward_p1(self, dL_dout: torch.Tensor):
        inputs = self.inputs.pop(0)
        e_x = torch.exp(-inputs)
        return dL_dout * ((1+e_x) + inputs*e_x)/((1+e_x)**2)


class llamaFF(Layer):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier = None, device:str = "cuda", dtype=torch.float32):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.device = device
        self.dtype = dtype
    
    @expose_params(acts=["l2", "silu_out"])
    def init_params(self):
        self.linears = [
            Dense(self.dim, self.hidden_dim, bias=False, device=self.device, dtype=self.dtype),
            Dense(self.hidden_dim, self.dim, bias=False, device=self.device, dtype=self.dtype),
            Dense(self.dim, self.hidden_dim, bias=False, device=self.device, dtype=self.dtype)
        ]
        for l in self.linears:
            l.init_params()
        
        self.silu = SiLU()

        self.l2 = []
        self.silu_out = []

    def forward(self, x: torch.Tensor):
        l0 = self.linears[0](x)
        l2 = self.linears[2](x)
        self.l2.append(l2)
        silu_out = self.silu(l0)
        self.silu_out.append(silu_out)
        return self.linears[1](silu_out*l2)
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout:torch.Tensor):
        dL_dl2in = self.linears[1].backward_p1(dL_dout)
        dL_dl0 = self.silu.backward_p1(dL_dl2in * self.l2.pop(0))
        dL_dx1 = self.linears[0].backward_p1(dL_dl0)
        dL_dx2 = self.linears[2].backward_p1(dL_dl2in * self.silu_out.pop(0)) 
        return dL_dx1 + dL_dx2


class RotaryEmbeddings(Layer):
    def __init__(self, dim, n_heads, seq_len, theta=10000.0, device:str="cuda"):
        super().__init__()
        dim = dim // n_heads
        self.device = device
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(seq_len, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.freqs_cis = self.freqs_cis.reshape([1, seq_len, 1, dim//2])
    
    def init_params(self):
        self.freqs_cis = self.freqs_cis.to(self.device)
    
    def forward(self, xq: torch.Tensor, xk: torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * self.freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * self.freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    def _per_grad_backpass(self, dL_dout):
        """
        Forward
        x = a + bi
        freq_cis = c + di
        x*freq_cis -> (a+bi)(c+di) = e +fi
        ac - bd + (ad +bc)i = e +fi 
        e = ac - bd & f = ad+bc

        Backward
        dL/da = dL/de * de/da + dL/df * df/da = c * dL/de + d * dL/df
        dL/db = dL/df * df/db - dL/de * de/da = c * dL/df - d dL/de
        """

        freq_cis = torch.view_as_real(self.freqs_cis)
        _dL_dout = dL_dout.float()
        dL_da = _dL_dout[:,:,:,:,0]*freq_cis[:,:,:,:,0] + _dL_dout[:,:,:,:,1]*freq_cis[:,:,:,:,1]
        dL_db = freq_cis[:,:,:,:,0]*_dL_dout[:,:,:,:,1] - freq_cis[:,:,:,:,1]*_dL_dout[:,:,:,:,0]
        return torch.stack([dL_da, dL_db], dim=-1).type_as(dL_dout)

    def backward_p1(self, dL_dxqout, dL_dxkout):
        #return self.forward(dL_dxqout, dL_dxkout)
        dL_dxqout_ = dL_dxqout.reshape(*dL_dxqout.shape[:-1], -1, 2)
        dL_dxkout_ = dL_dxkout.reshape(*dL_dxkout.shape[:-1], -1, 2)
        dL_dxq = self._per_grad_backpass(dL_dxqout_).flatten(3)
        dL_dxk = self._per_grad_backpass(dL_dxkout_).flatten(3)
        return dL_dxq, dL_dxk

        
class GroupedMultiQueryAttention(Layer):
    def __init__(self, emb_dim: int, num_heads: int, num_kv_heads:int, max_seq_len:int, theta:float=10000.0, p:float=0.0, device:str="cuda", dtype=torch.float32):
        super().__init__()
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = emb_dim // num_heads
        self.theta = theta
        self.p = p
        self.inv_square_d = 1/torch.sqrt(torch.tensor(self.head_dim, device=device))

        self.device = device
        self.dtype = dtype

    @expose_params(acts=["xq", "xk", "xv", "sQK_T"])
    def init_params(self):
        self.linears = {"Q": Dense(self.emb_dim, self.num_heads*self.head_dim, bias=False, device=self.device, dtype=self.dtype),
                        "K": Dense(self.emb_dim, self.num_kv_heads*self.head_dim, bias=False, device=self.device, dtype=self.dtype),
                        "V": Dense(self.emb_dim, self.num_kv_heads*self.head_dim, bias=False, device=self.device, dtype=self.dtype),
                        "O": Dense(self.num_heads*self.head_dim, self.emb_dim, bias=False, device=self.device, dtype=self.dtype)
                        }
        for l in self.linears.keys():
            self.linears[l].init_params()
        
        self.dropouts = {"Q": Dropout(p=self.p),
                         "K": Dropout(p=self.p),
                         "V": Dropout(p=self.p)
                        }

        self.softmax_fn = Softmax()
        
        self.rotary_emb = RotaryEmbeddings(self.emb_dim, self.num_heads, self.max_seq_len, theta=self.theta, device=self.device)
        self.rotary_emb.init_params()

        self.xq = []
        self.xk = []
        self.xv = []
        self.sQK_T = []
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seqlen, _ = x.shape

        xq = self.dropouts["Q"](self.linears["Q"](x))
        xk = self.dropouts["K"](self.linears["K"](x))
        xv = self.dropouts["V"](self.linears["V"](x))

        xq = xq.view(batch_size, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seqlen, self.num_kv_heads, self.head_dim)
        self.xv.append(xv)


        xq, xk = self.rotary_emb.forward(xq, xk)
        xq = torch.cat(xq.unsqueeze(-3).chunk(self.n_rep, dim=-2), dim=-3)  # broadcast for memory efficiency

        self.xq.append(xq)
        self.xk.append(xk)
        #self.xq = xq #torch.chunk(xq, self.n_rep, dim=-2)

        QK_T = vmap(lambda q, k: vmap(lambda _q: torch.bmm(_q, k.transpose(-1,-2)), in_dims=-2, out_dims=-2)(q), in_dims=-2, out_dims=-2)(xq, xk) * self.inv_square_d
        QK_T = torch.flatten(QK_T, -3, -2)
        if mask is not None:
            QK_T = vmap(lambda x: x + mask, in_dims=-2, out_dims=-2)(QK_T)
        sQK_T = self.softmax_fn(QK_T)
        sQK_T = torch.cat(sQK_T.unsqueeze(-3).chunk(self.n_rep, dim=-2), dim=-3)

        out = vmap(lambda sqkt, v: vmap(lambda _sqkt: torch.bmm(_sqkt, v), in_dims=-2, out_dims=-2)(sqkt), in_dims=-2, out_dims=-2)(sQK_T, xv)
        sQK_T = sQK_T.flatten(-3,-2)
        self.sQK_T.append(sQK_T)
        
        out = torch.flatten(out, -3, -2)
        out = torch.flatten(out, -2, -1)
        return self.linears["O"](out)

    @multi_stage_wrapper
    def backward_p1(self, dL_dout: torch.tensor):
        #vmap go fast
        #with nvtx_profile("output_linear"):
        dL_dAtt = self.linears["O"].backward_p1(dL_dout)
        dL_dAtt = torch.cat(dL_dAtt.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        dL_dAtt = torch.cat(dL_dAtt.unsqueeze(-3).chunk(self.n_rep, dim=-2), dim=-3)  # broadcast for memory efficiency

        #with nvtx_profile("sqk back"):
        dL_dsQKT = vmap(lambda dldatt, v: vmap(lambda _dldatt: torch.bmm(_dldatt, v.transpose(-1,-2)), in_dims=-2, out_dims=-2)(dldatt), in_dims=-2, out_dims=-2)(dL_dAtt, self.xv.pop(0))
        dL_dsQKT = torch.flatten(dL_dsQKT, -3, -2)

        #with nvtx_profile("softmax"):
        dL_dQKT = self.softmax_fn.backward_p1(dL_dsQKT) * self.inv_square_d
        
        dL_dQKT =  torch.cat(dL_dQKT.unsqueeze(-3).chunk(self.n_rep, dim=-2), dim=-3)

        #with nvtx_profile("q back"):
        dL_dxq = vmap(lambda dldqkt, k: vmap(lambda _dldqkt: torch.bmm(_dldqkt, k), in_dims=-2, out_dims=-2)(dldqkt), in_dims=-2, out_dims=-2)(dL_dQKT, self.xk.pop(0))
        dL_dxq = dL_dxq.flatten(-3,-2)

        #with nvtx_profile("k back"):
        dL_dxkt = vmap(vmap(lambda q, dldqkt: torch.bmm(q.transpose(-1,-2), dldqkt), in_dims=-2, out_dims=-2), in_dims=-2, out_dims=-2)(self.xq.pop(0), dL_dQKT)
        dL_dxkt = torch.sum(dL_dxkt, -3)

        dL_dxk = torch.vmap(lambda dl_dlkt: torch.transpose(dl_dlkt, -1, -2), in_dims=-2, out_dims=-2)(dL_dxkt)

        sQK_T = torch.cat(self.sQK_T.pop(0).unsqueeze(-3).chunk(self.n_rep, dim=-2), dim=-3)
        #with nvtx_profile("v back"):
        dL_dxv = vmap(vmap(lambda sqkt, dldatt: torch.bmm(sqkt.transpose(-1,-2), dldatt), in_dims=-2, out_dims=-2), in_dims=-2, out_dims=-2)(sQK_T, dL_dAtt)
        dL_dxv = torch.sum(dL_dxv, -3)
        
        #with nvtx_profile("RotaryEmbeddings"):
        dL_dxq, dL_dxk = self.rotary_emb.backward_p1(dL_dxq, dL_dxk)

        #ƒwith nvtx_profile("Linears"):
        dL_dQ = self.linears["Q"].backward_p1(self.dropouts["Q"].backward_p1(torch.flatten(dL_dxq, -2, -1)))
        dL_dK = self.linears["K"].backward_p1(self.dropouts["K"].backward_p1(torch.flatten(dL_dxk, -2, -1)))
        dL_dV = self.linears["V"].backward_p1(self.dropouts["V"].backward_p1(torch.flatten(dL_dxv, -2, -1)))

        return dL_dQ + dL_dK + dL_dV

    

# Transformer++
class TransformerPPBlock(Layer):
    def __init__(self, dim, num_heads, num_kv_heads, max_seqlen, theta=10000.0, p=0.1, device="cuda", dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seqlen = max_seqlen
        self.theta = theta
        self.p = p

        self.device = device
        self.dtype = dtype
    
    def init_params(self):
        self.attention = GroupedMultiQueryAttention(self.dim, self.num_heads, self.num_kv_heads, self.max_seqlen, theta=self.theta, p=self.p, device=self.device, dtype=self.dtype)
        self.ff = llamaFF(self.dim, self.dim*4, 256, device=self.device, dtype=self.dtype)
        self.att_norm = NLPRMSNorm(-1, self.dim, device=self.device, dtype=self.dtype)
        self.ff_norm = NLPRMSNorm(-1, self.dim, device=self.device, dtype=self.dtype)
        self.att_dropout = Dropout(self.p)
        self.ff_dropout = Dropout(self.p)

        mask = torch.zeros(self.max_seqlen, self.max_seqlen, device=self.device, dtype=self.dtype)
        self.mask = mask.masked_fill(torch.triu(torch.ones(self.max_seqlen, self.max_seqlen, device=self.device, dtype=self.dtype), diagonal=1).bool(), float('-inf'))
    
        self.attention.init_params()
        self.ff.init_params()
        self.att_norm.init_params()
        self.ff_norm.init_params()

    def forward(self, x):
        n_x = self.att_norm(x)
        if self.training:
            x = self.attention(n_x, self.mask) + x
        else:
            x = self.attention(n_x) + x
        h = self.att_dropout(x)
        x = self.ff(self.ff_norm(h)) + h
        return self.ff_dropout(x)

    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        #with nvtx_profile("ff_dropout"):
        dL_dff = self.ff_dropout.backward_p1(dL_dout)
        #with nvtx_profile("ff_back"):
        dL_h = self.ff_norm.backward_p1(self.ff.backward_p1(dL_dff)) + dL_dff
        #with nvtx_profile("att_dropout"):
        dL_datt = self.att_dropout.backward_p1(dL_h)
        dL_dnorm = self.attention.backward_p1(dL_datt)
        #with nvtx_profile("att_norm"):
        out = self.att_norm.backward_p1(dL_dnorm) + dL_datt
        return out

class BilinearUpSample(Layer):
    def __init__(self, factor):
        self.factor = 2
    
    def forward(self, x):
        return F.upsample_bilinear(x)

class DiffUpsample(Layer):
    def __init__(self, in_channels, conv=False):
        super().__init__()
        self.in_channels = in_channels
        self.conv = conv
    
    def init_params(self):
        self.conv0 = Conv2D(self.in_channels, self.in_channels, 3, padding=1, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.conv:
            x = self.conv0(x)
        return x
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        if self.conv:
            dL_dout = self.conv0.backward_p1(dL_dout)
        return F.avg_pool2d(dL_dout, 2, divisor_override=1)

class DiffDownsample(Layer):
    def init_params(self):
        self.pool = AvgPool2D(2)
    def forward(self,x):
        return self.pool(x)
    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        return self.pool.backward_p1(dL_dout)

class DiffResnetBlock(Layer):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.dropout = dropout

    def init_params(self):
        self.bn1 = BatchNorm2D(self.in_channels)
        self.conv1 = Conv2D(self.in_channels, self.out_channels, 3, padding=1, bias=False)
        self.swish_1 = SiLU()
        
        self.bn1.init_params()
        self.conv1.init_params()
        
        self.bn2 = BatchNorm2D(self.out_channels)
        self.dropout = Dropout(self.dropout)
        self.conv2 = Conv2D(self.out_channels, self.out_channels, 3, padding=1)
        self.swish_2 = SiLU()
        
        self.bn2.init_params()
        self.conv2.init_params()
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv2D(self.in_channels, self.out_channels, 3, padding=1)
                self.conv_shortcut.init_params()
            else:
                self.nin_shortcut = Conv2D(self.in_channels, self.out_channels, 1, padding=0)
                self.nin_shortcut.init_params()
                
    def forward(self, x):
        h = x
        h = self.bn1(h)
        h = self.swish_1(h)
        h = self.conv1(h)
        
        h = self.bn2(h)
        h = self.swish_2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        
        return x+h
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        dL_dh = self.conv2.backward_p1(dL_dout)
        dL_dh = self.dropout.backward_p1(dL_dh)
        dL_dh = self.swish_2.backward_p1(dL_dh)
        dL_dh = self.bn2.backward_p1(dL_dh)
        dL_dh = self.conv1.backward_p1(dL_dh)
        dL_dh = self.swish_1.backward_p1(dL_dh)
        dL_dh = self.bn1.backward_p1(dL_dh)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                dL_dx = self.conv_shortcut.backward_p1(dL_dh)
            else:
                dL_dx = self.nin_shortcut.backward_p1(dL_dh)
        else:
            dL_dx = dL_dout

        return dL_dh + dL_dx
    
        
class DiffAttention(Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
    
    @expose_params(acts=["q", "k", "v","sqkt"])
    def init_params(self):
        self.bn = BatchNorm2D(self.in_channels)
        self.bn.init_params()

        self.conv_q = Conv2D(self.in_channels, self.in_channels, 1, padding=0)
        self.conv_k = Conv2D(self.in_channels, self.in_channels, 1, padding=0)
        self.conv_v = Conv2D(self.in_channels, self.in_channels, 1, padding=0)
        self.conv_out = Conv2D(self.in_channels, self.in_channels, 1, padding=0)

        self.conv_q.init_params()
        self.conv_k.init_params()
        self.conv_v.init_params()
        self.conv_out.init_params()
        
        self.softmax = Softmax()
        
        self.inv_sqrt_d = self.in_channels**(-0.5)
        
        self.q = []
        self.k = []
        self.v = []
        self.sqkt = []
    
    def forward(self, x:torch.Tensor):
        h_ = x
        h_ = self.bn(h_)
        q = self.conv_q(h_)
        k = self.conv_k(h_)
        v = self.conv_v(h_)
        
        b,c,h,w = q.shape
        q = q.reshape(b,c,-1)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, -1)
        
        self.q.append(q)
        self.k.append(k)
        
        qk_t = torch.bmm(q, k)*self.inv_sqrt_d
        
        sqk_t = self.softmax(qk_t)
        self.sqkt.append(sqk_t)
        v = v.reshape(b, c, -1)
        v = v.permute(0, 2, 1)
        self.v.append(v)
        h_ = torch.bmm(v, sqk_t)
        h_ = h_.reshape(b, c, h, w)
        
        h_ = self.conv_out(h_)
        
        return x + h_
    
    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        b,c,h,w = dL_dout.shape
        dL_datt = self.conv_out.backward_p1(dL_dout)
        dL_datt = dL_dout.reshape(b,c,-1)
        dL_dsqkt = torch.bmm(torch.transpose(self.v.pop(0), -2, -1), dL_datt)
        dL_dqkt = self.softmax.backward_p1(dL_dsqkt)
        dL_dv = torch.bmm(dL_datt, torch.transpose(self.sqkt.pop(0), -2, -1))
        dL_dq = torch.bmm(dL_dqkt, torch.transpose(self.k.pop(0), -2, -1))
        dL_dk = torch.bmm(torch.transpose(self.q.pop(0), -2, -1), dL_dqkt)
        
        dL_dq = dL_dq.permute(0, 2, 1)
        dL_dq = self.conv_q.backward_p1(dL_dq.reshape(b, c, h, w))
        dL_dk = self.conv_k.backward_p1(dL_dk.reshape(b, c, h, w))
        dL_dv =self.cnv_v.backward_p1(dL_dv.reshape(b, c, h, w))
        
        dL_dh = dL_dq + dL_dk + dL_dv
        dL_dh = self.bn.backward_p1(dL_dh)
        
        return dL_dh + dL_dout

class DiffUnet(Layer):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=False, in_channels,
                 resolution, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        #Down
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = []
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(DiffResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(DiffAttention(block_in))
            down = {
                "block":block,
                "attn":attn
            }
            if i_level != self.num_resolutions-1:
                down["downsample"] = DiffDownsample(block_in, dropout=dropout)
                curr_res = curr_res // 2
            self.down.append(down)
        
        #Mid
        self.mid = {
            "block_1": DiffResnetBlock(block_in, block_in, dropout=dropout),
            "attn_1": DiffAttention(block_in),
            "block_2": DiffResnetBlock(block_in, block_in, dropout=dropout)
        }

        #Up
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(DiffResnetBlock(in_channels=block_in+skip_in,
                                             out_channels=block_out,
                                             dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(DiffAttention(block_in))
            up = {
                "block":block,
                "attn":attn
            }
            if i_level != 0:
                up["upsample"] = DiffUpsample(block_in, dropout=dropout)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        
        #Out
        self.bn_out = BatchNorm2D(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        

class SelectiveScan(Layer):
    def __init__(self, delta_softplus=False):
        super().__init__()
        self.delta_softplus = delta_softplus
    
    @expose_params(acts=["u", "delta", "A", "B", "C", "D", "z", "delta_bias", "x", "out"])
    def init_params(self):
        self.u = []
        self.delta = []
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.z = []
        self.delta_bias = []
        self.x = []
        self.out = []
    
    def _save_params(self, u, delta, A, B, C, D, z, delta_bias, x, out):
        self.u.append(u)
        self.delta.append(delta)
        self.A.append(A)
        self.B.append(B)
        self.C.append(C)
        self.D.append(D)
        self.z.append(z)
        self.delta_bias.append(delta_bias)
        self.x.append(x)
        self.out.append(out)
    
    #@cleanup_act("u", "delta", "A", "B", "C", "D", "z", "delta_bias", "x", "out")
    def _load_parms(self):
        u = self.u.pop(0)
        delta = self.delta.pop(0)
        A = self.A.pop(0)
        B = self.B.pop(0)
        C = self.C.pop(0)
        D = self.D.pop(0)
        z = self.z.pop(0)
        delta_bias = self.delta_bias.pop(0)
        x = self.x.pop(0)
        out = self.out.pop(0)
        return u, delta, A, B, C, D, z, delta_bias, x, out
    
    def forward(self, u, delta, A, B, C, D=None, z=None, delta_bias=None):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
    
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, self.delta_softplus)
        
        self._save_params(u, delta, A, B, C, D, z, delta_bias, x, out)
        
        return rest[0]
    
    def backward_p1(self, dL_dout):
        u, delta, A, B, C, D, z, delta_bias, x, out = self._load_parms()
        if dL_dout.stride(-1) != 1:
            dL_dout = dL_dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(u, delta, A, B, C, D, z, delta_bias, dL_dout, x, out, None, self.delta_softplus, False)

        dz = rest[0]
        
        dB = dB.squeeze(1) 
        dC = dC.squeeze(1)
        
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, dz
        

class Mamba(Layer):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        layer_idx=None,
        device="cuda",
        dtype=torch.float32
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.layer_idx = layer_idx
        self.device = device
        self.dtype = dtype
        
        self.act = SiLU()
        
        self.x_proj = Dense(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, device=device, dtype=self.dtype)
        
        self.dt_init_std = self.dt_rank**-0.5 * dt_scale

        self.out_proj = Dense(self.d_inner, self.d_model, bias=bias, device=self.device, dtype=self.dtype)
    
    @expose_params({"conv_kernel":"conv_kernel_grad", 
                    "conv_bias":"conv_bias_grad", 
                    "dt_proj_weight":"dt_proj_weight_grad", 
                    "dt_proj_bias":"dt_proj_bias_grad", "D":"D_grad",
                    "in_proj_weight":"in_proj_weight_grad",
                    "in_proj_bias":"in_proj_bias_grad",
                    "D":"D_grad",
                    "A_log":"A_log_grad"},
                   ["dL_dxz", "x", "conv_in", "dL_dact", "dL_dtt", "dt_t", "dA"])
    def init_params(self):
        self.scan = SelectiveScan(True)
        self.scan.init_params()

        self.in_proj_weight, self.in_proj_bias = init_params.dense_params(self.d_model, self.d_inner*2, factory_kwargs={"device": self.device, "dtype":self.dtype})
        self.in_proj_weight = self.in_proj_weight.T.contiguous()
        self.in_proj_bias = self.in_proj_bias.unsqueeze(-1)
        self.in_proj_weight_grad = torch.zeros_like(self.in_proj_weight)
        self.in_proj_bias_grad = torch.zeros_like(self.in_proj_bias)        
        
        self.conv_kernel, self.conv_bias = init_params.mamba_conv1d_params(self.d_inner, self.d_conv, factory_kwargs={"device": self.device, "dtype":self.dtype})

        self.conv_kernel_grad = torch.zeros_like(self.conv_kernel)
        self.conv_bias_grad = torch.zeros_like(self.conv_bias)
        
        m = torch.distributions.uniform.Uniform(-self.dt_init_std, self.dt_init_std)
        self.dt_proj_weight = m.sample((self.d_inner, self.dt_rank)).to(self.device).to(self.dtype)
        self.dt_proj_weight_grad = torch.zeros_like(self.dt_proj_weight)
        
        
        dt = torch.exp(
            torch.rand(self.d_inner, device=self.device, dtype=self.dtype) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)
        
        self.dt_proj_bias = dt + torch.log(-torch.expm1(-dt)) 
        self.dt_proj_bias_grad = torch.zeros_like(self.dt_proj_bias)
        
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        
        self.A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log_grad = torch.zeros_like(self.A_log)
        self.D = torch.ones(self.d_inner, device=self.device)  # Keep in fp32
        self.D_grad = torch.zeros_like(self.D)

        self.x_proj.init_params()
        self.out_proj.init_params()

        self.x = []
        self.conv_in = []
        self.dt_t = []
        self.dL_dtt = []
        self.dL_dxz = []
        self.dL_dact = []
        self.dA = []
    
    
    def forward(self, x:torch.Tensor):

        batch, seqlen, dim = x.shape

        x = rearrange(x, "b l d -> d (b l)")
        self.x.append(x)

        xz = rearrange(
            self.in_proj_weight @ x,
            "d (b l) -> b d l",
            l=seqlen,
        )

        xz = xz + self.in_proj_bias

        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=1)
        self.conv_in.append(x)
        x = F.conv1d(x, self.conv_kernel, self.conv_bias, groups=self.d_inner, padding=self.d_conv - 1)
        x = x[..., :seqlen]
        x = self.act(x)
        

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_t = dt.t()
        self.dt_t.append(dt_t)
        dt = self.dt_proj_weight @ dt_t

        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen)
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen)

        y = self.scan(x, dt, A, B, C, self.D.float(), z=z, delta_bias=self.dt_proj_bias.float())

        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

    @multi_stage_wrapper
    def backward_p1(self, dL_dout:torch.Tensor):
        batch, seqlen, dim = dL_dout.shape
        dL_dy = self.out_proj.backward_p1(dL_dout)
        dL_dy = rearrange(dL_dy, "b l d -> b d l")

        du, ddelta, dA, dB, dC, dD, ddelta_bias, dz = self.scan.backward_p1(dL_dy)
        self.dA.append(dA)
        self.dt_proj_bias_grad[:] += ddelta_bias
        self.D_grad[:] += dD

        dL_dC = rearrange(dC, "b dstate l -> (b l) dstate")
        dL_dB = rearrange(dB, "b dstate l -> (b l) dstate")
        dL_dt = rearrange(ddelta, "b d l -> d (b l)")
        self.dL_dtt.append(dL_dt)
        dL_dt_t = self.dt_proj_weight.T @ dL_dt
        dL_dt = dL_dt_t.t()
        dL_dxdbl = torch.cat([dL_dt, dL_dB, dL_dC], dim=-1)
        #print(dL_dxdbl.shape)
        dL_dx = self.x_proj.backward_p1(dL_dxdbl)
        dL_dx = rearrange(dL_dx, "(b l) d -> b d l", l=seqlen) + du
        dL_dx = self.act.backward_p1(dL_dx)
        self.dL_dact.append(dL_dx)
        dL_dx = F.pad(dL_dx, (0, self.d_conv - 1))
        dL_dx = F.conv1d(dL_dx, self.conv_kernel.flip(-1), groups=self.d_inner)

        dL_dxz = torch.cat((dL_dx, dz), dim=1)
        dL_dxz = rearrange(dL_dxz, "b d l -> d (b l)")
        self.dL_dxz.append(dL_dxz)

        dL_dx = self.in_proj_weight.T @ dL_dxz
        dL_dx = rearrange(dL_dx, "d (b l) -> b l d", l=seqlen)

        return dL_dx

    @cleanup_act("dL_dxz", "x", "conv_in", "dL_dact", "dL_dtt", "dt_t", "dA")
    def backward_p2(self, inter=False):
        #n = len(self.dL_dxz) if not inter else 1
        #for _ in range(n):
        if self.multi_stage and not inter:
            dL_dxz = torch.stack(self.dL_dxz)
            x = torch.stack(self.x)
            conv_in = torch.cat(self.conv_in)
            dL_dact = torch.cat(self.dL_dact)
            dL_dtt = torch.stack(self.dL_dtt)
            dt_t = torch.stack(self.dt_t)
            dA = torch.stack(self.dA)
        else: 
            dL_dxz = self.dL_dxz.pop(0).unsqueeze(0)
            x = self.x.pop(0).unsqueeze(0)
            conv_in = self.conv_in.pop(0)
            dL_dact = self.dL_dact.pop(0)
            dL_dtt = self.dL_dtt.pop(0).unsqueeze(0)
            dt_t = self.dt_t.pop(0).unsqueeze(0)
            dA = self.dA.pop(0).unsqueeze(0)
        
        self.in_proj_weight_grad[:] += torch.sum(dL_dxz @ x.mT, dim=0)
        self.in_proj_bias_grad[:] += torch.sum(dL_dxz, dim=(0,2)).unsqueeze(-1)
    
        self.conv_bias_grad[:] += torch.sum(dL_dact, dim=(0,2))
        conv_in = F.pad(conv_in, (self.d_conv - 1,  0))
        # Witchcraft
        self.conv_kernel_grad[:] += torch.sum(vmap(vmap(F.conv1d,in_dims=(1,0)))(conv_in.unsqueeze(1).unsqueeze(-2), dL_dact.unsqueeze(-2).unsqueeze(-2)), dim=(0, -2))

        self.dt_proj_weight_grad[:] += torch.sum(dL_dtt @ dt_t.mT, dim=0)
        self.A_log_grad[:] = torch.sum(dA * -torch.exp(self.A_log.float()), dim=0)

class MambaBlock(Layer):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        layer_idx=None,
        device="cuda", 
        dtype=torch.float32
    ):  
        super().__init__()
        self.mamba = Mamba(d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max, dt_init, dt_scale, dt_init_floor, bias, layer_idx, device, dtype)
        self.norm = NLPRMSNorm(-1, d_model, device=device, dtype=dtype)
    
    def init_params(self):
        self.mamba.init_params()
        self.norm.init_params()
    
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        return hidden_states + residual

    @multi_stage_wrapper
    def backward_p1(self, dL_dout):
        dL_dnorm = self.mamba.backward_p1(dL_dout)
        dL_din = self.norm.backward_p1(dL_dnorm)
        return dL_din + dL_dout


if __name__ == "__main__":
    def test(layer: Layer, x, dL_dout):
        import random
        print(layer.__class__.__name__)
        layer.clear_acts()
        layer.init_params()
        with torch.no_grad():
            layer.forward(x)
            my_back = layer.backward_p1(dL_dout)
            #layer.backward_p2(inter=False)
        torch.manual_seed(0)
        true_back = torch.autograd.functional.vjp(layer.forward, x, dL_dout)[1]
        idxs = [random.randint(0, true_back.flatten().shape[0]) for _ in range(20)]
        print("-"*40)
        print("Predicted Grads")
        print(my_back.flatten()[idxs])
        print("-"*40)
        print("True Grads")
        print(true_back.flatten()[idxs])
        print("-"*40)
        print("        Predicted          True")
        print("Means: ", my_back.mean().item(), true_back.mean().item())
        print("Stds : ", my_back.std().item(), true_back.std().item())
        num_correct = (torch.isclose(my_back, true_back, rtol=0.0001)).sum()
        print(f"Num correct: {num_correct}/{my_back.numel()} ({100*num_correct/my_back.numel():.2f}%)")
        return my_back, true_back
    
    def test_dropout():
        test(Dropout(0.1), torch.randn(16,24,80), torch.ones(16,24,80))

    
    def test_multihead():
        x = torch.randn(16, 24, 80)
        dL_dout = torch.ones(16,24,80)
        layer = MultiHeadAttention(80, 8, p=0, device="cpu")
        with torch.no_grad():
            layer.forward(x, x, x)
            my_back = torch.sum(torch.stack(layer.backward_p1(dL_dout)), dim=0)
            layer.backward_p2()
        true_back = torch.autograd.functional.vjp(lambda _x: layer.forward(_x, _x, _x), x, dL_dout)[1]
        print(my_back[0,0])
        print(true_back[0,0])
        print(torch.allclose(my_back, true_back))
    
    def test_grouped_mulit_head():
        x = torch.randn(16, 30, 80)
        dL_dout = torch.ones(16,30,80)
        layer = GroupedMultiQueryAttention(80, 8, 4, 30, device="cpu")
        layer.init_params()
        with torch.no_grad():
            layer.forward(x)
            my_back = layer.backward_p1(dL_dout)
            layer.backward_p2()
        true_back = torch.autograd.functional.vjp(layer.forward, x, dL_dout)[1]
        print(my_back[0,0])
        print(true_back[0,0])
        num_correct = (torch.isclose(my_back, true_back, rtol=0.00001)).sum()
        print(f"Num correct: {num_correct}/{my_back.numel()} ({100*num_correct/my_back.numel():.2f}%)")
        print(torch.allclose(my_back, true_back))
    
    def test_relu():
        x = torch.randn(16, 24, 80)
        dL_dout = torch.ones(16, 24, 80)
        layer = ReLU()
        test(layer, x, dL_dout)
    
    def test_layernorm():
        m = torch.distributions.Uniform(1, 3)
        test(NLPLayerNorm(-1, 80), m.sample((16,24,80)), torch.ones(16,24,80))
    
    def test_bert():
        m = torch.distributions.Uniform(1, 3)
        x = torch.randn(16, 24, 80)
        x = m.sample(x.shape)
        dL_dout = torch.ones(16, 24, 80)
        layer = BertBlock(80, 8, 160, p=0, device="cpu")
        test(layer, x, dL_dout)
    
    def test_conv2D():
        image = torch.randn(16, 3, 80, 80, device="cuda")
        conv = Conv2D(3, 16, 3)
        test(conv, image, torch.ones(16,16,78,78, device="cuda"))
    
    def test_batchnorm():
        m = torch.distributions.Uniform(1, 3)
        image = m.sample((16, 3, 80, 80)).to("cuda").to(torch.float64)
        test(BatchNorm2D(3, dtype=torch.float64), image, torch.ones(16,3,80,80, device="cuda", dtype=torch.float64))

    def test_dense():
        x = torch.randn(16, 80, device="cuda")
        test(Dense(80, 160), x, torch.ones(16,160, device="cuda"))
    
    def test_resnet():
        image = torch.randn(16, 3, 32, 32)
        resnet50 = ResNet(ResNetBottleneck, [3, 4, 6, 3], device="cpu")
        resnet50.init_params()
        test(resnet50, image, torch.ones_like(resnet50(image)))
    
    def test_rmsnorm():
        m = torch.distributions.Uniform(1, 3)
        x = m.sample((16,24,80))
        dL_dout = torch.ones(16,24,80)
        layer = NLPRMSNorm(-1,80, device="cpu")
        test(layer, x, dL_dout)
    
    def test_rotary():
        xq = torch.randn(16, 24, 8, 10)
        dL_dout = torch.ones(16,24,8,10)
        model = RotaryEmbeddings(80, 8, 24)
        with torch.no_grad():
            model.forward(xq, xq)
            my_back = torch.sum(torch.stack(model.backward_p1(dL_dout, dL_dout)), dim=0)
            model.backward_p2()
        true_back = torch.autograd.functional.vjp(lambda _x: model.forward(_x, _x), xq, (dL_dout, dL_dout))[1]
        print(torch.allclose(my_back, true_back))

    def test_transformer_pp():
        x = torch.randn(16, 24, 80, device="cpu")
        dL_dout = torch.ones(16,24,80, device="cpu")
        layer = TransformerPPBlock(80, 8, 4, 24, device="cpu")
        test(layer, x, dL_dout)
    
    def test_llamaff():
        x = torch.randn(16, 24, 80, device="cuda")
        dL_dout = torch.ones(16,24,80, device="cuda")
        layer = llamaFF(80, 160, 160, device="cuda")
        test(layer, x, dL_dout)
    
    def test_softmax():
        x = torch.randn(16,24,8,24)
        dL_dout = torch.ones(16,24,8,24)
        layer = Softmax()
        test(layer, x, dL_dout)
    
    def test_avg_pool():
        x = torch.randn(16,512,28,28)
        dL_dout = torch.randn(16,512,1,1)
        layer = AvgPool2D(28)
        test(layer, x, dL_dout)

    def test_max_pool():
        x = torch.randn(16, 512, 28, 28)
        dL_dout = torch.randn(16, 512, 1, 1)
        layer = MaxPool2D(28)
        test(layer, x, dL_dout)

    def test_mamba():
        torch.manual_seed(1)
        x = torch.randn(4, 32 , 64, device="cuda")
        dL_dout = torch.ones(4, 32 , 64, device="cuda")
        layer = MambaBlock(64)
        test(layer, x, dL_dout)

if __name__ == "__main__":
    test_max_pool()
