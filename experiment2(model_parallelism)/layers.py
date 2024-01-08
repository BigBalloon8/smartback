from typing import Any, Dict, Optional, Union, Sequence, Tuple
from abc import ABC, abstractmethod
from functools import wraps

import torch
import torch.nn.functional as F
from torch import vmap as vmap

class cleanup(object):
    def __init__(self, *args):
        self.args = args
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            for arg in self.args:
                setattr(args[0], arg, None)
        return wrapper

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

class Activation(Layer):
    """
    Used for Activation functions that dont have parameters
    """
    def backward_p2(self):
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
            self.streams = []
            for _ in self.layers:
                self.streams.append(torch.cuda.Stream())
        else:
            self.device = "cpu"
        
    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)
    
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward_p1(self, dL_dout: torch.Tensor):
        for layer in self.layers[::-1]:
            dL_dout = layer.backward_p1(dL_dout)
        return dL_dout
    
    def backward_p2(self):
        if self.device == "cuda":
            for i, layer in enumerate(self.layers):
                with torch.cuda.stream(self.streams[i]):
                    layer.backward_p2()
        else:
            for layer in self.layers:
                layer.backward_p2()
    


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        """
        Initialize a Dense Layer

        Args:
            input_size (int): size of input vector
            output_size (int): size of output vector
            bias (bool): whether or not to include bias
        
        Attributes:
            weights (torch.Tensor): weights of the dense layer
            bias (torch.Tensor): bias of the dense layer
            weights_g (torch.Tensor): gradients of the weights of the dense layer
            bias_g (torch.Tensor): gradients of the bias of the dense layer
            inputs (torch.Tensor): inputs of the dense layer
            dL_dout (torch.Tensor): derivative of the loss with respect to the dense layers output
        
        """
        m = torch.distributions.normal.Normal(0, 0.01)
        self.weights = m.sample(torch.Size([input_size, output_size]))
        self.bias = torch.zeros(output_size) if bias else None
        
        self.weights_g = torch.zeros_like(self.weights)
        self.bias_g = torch.zeros_like(self.bias) if bias else None
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the dense layer

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after applying dense layer
        """
        self.inputs = x
        x =  vmap(lambda x_: torch.matmul(x_, self.weights))(x)
        if self.bias is not None: 
            out = torch.add(x, self.bias)
        else:
            out = x
        return out
    
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the dense layers output and returns the derivative of the loss with respect to the dense layers input

        Args:
            dL_dout (torch.Tensor): Derivative of the loss with respect to the dense layers output

        Returns:
            torch.Tensor: Derivative of the loss with respect to the dense layers input
        """
        self.dL_dout = dL_dout
        return vmap(lambda dl_dout: torch.matmul(dl_dout, self.weights.T))(dL_dout)
    
    @cleanup("inputs", "dL_dout")
    def backward_p2(self):
        """
        Calculates the gradients of the dense layer's parameters
        """
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
        #self.inputs = None
        
        if self.bias is not None:
            self.bias_g[:] = torch.sum(self.dL_dout, dim=tuple(range(self.dL_dout.ndim)[:-1]))
        
        #self.dL_dout = None


class ReLU(Activation):
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the relu

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after applying relu
        """
        self.inputs = x
        out = torch.maximum(x, torch.tensor(0.0, dtype=x.dtype))
        return out
    
    @cleanup("inputs")
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the relus output and returns the derivative of the loss with respect to the relus input

        Args:
            dL_dout (torch.Tensor): Derivative of the loss with respect to the dense layers output

        Returns:
            torch.Tensor: Derivative of the loss with respect to the dense layers input
        """
        dout_din = torch.where(self.inputs>0, 1.0, 0.0)
        return dL_dout*dout_din

    def backward_p2(self):
        pass
    
class Dropout(Layer):
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
        self.p = p
        self.training = True
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the dropout layer. 
        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output tensor after applying the dropout
        """
        torch.manual_seed(0)
        if not self.training or self.p==0:
            return x
        if self.p==1:
            return torch.zeros_like(x)
        self.p_mask = torch.bernoulli(torch.ones_like(x) -self.p)
        return x*self.p_mask
    
    @cleanup("p_mask")
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
        return dL_dout*self.p_mask

    
    def backward_p2(self):
        pass

    
class MultiHeadAttention(Layer):
    #TODO self is the first layer in the model you can cache the linear outputs of the first 3 linears
    def __init__(self, emb_dim: int, num_heads: int, p: Optional[float]=0.1, device="cuda"):
        """
        Initializes the multihead attention

        Args:
            emb_dim (int): Size of the embedding dimension
            num_heads (int): Number of attention heads
            p (Optional[float]): Probability for the dropouts. Defaults to 0.1.
        
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
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.inv_sqrt_d = (emb_dim / num_heads)**(-1/2)
        self.linears = {"Q": Dense(self.emb_dim, self.emb_dim),
                        "K": Dense(self.emb_dim, self.emb_dim),
                        "V": Dense(self.emb_dim, self.emb_dim),
                        "O": Dense(self.emb_dim, self.emb_dim)
                        }
        self.dropouts = {"Q": Dropout(p=p),
                         "K": Dropout(p=p),
                         "V": Dropout(p=p)
                        }
        
        self.inputs = {"Q": None, "K": None, "V": None}
        
        if "cuda" in device:
            self.device = "cuda" 
            self.streams = []
            for _ in range(len(self.linears)):
                self.streams.append(torch.cuda.Stream())
        else:
            self.device = "cpu"
    
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
        self.inputs["Q"] = Q
        self.inputs["K"] = K
        self.inputs["V"] = V
        
        lQ = self.dropouts["Q"](self.linears["Q"](Q))
        lK = self.dropouts["K"](self.linears["K"](K))
        lV = self.dropouts["V"](self.linears["V"](V))
        
        self.lQ = torch.cat(lQ.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        self.lK = torch.cat(lK.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        self.lV = torch.cat(lV.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)
        
        QK_T = vmap(lambda q, k: torch.bmm(q, torch.transpose(k, -1, -2)), in_dims=-2, out_dims=-2)(self.lQ, self.lK) * self.inv_sqrt_d
        if mask is not None:
            QK_T = vmap(lambda x: x + mask, in_dims=-2, out_dims=-2)(QK_T)
        def _softmax(x):
            e_x = torch.exp(x)
            return e_x / torch.sum(e_x)
        self.sQK_T = vmap(vmap(vmap(_softmax)))(QK_T)
        #torch.softmax(QK_T, dim=-1, out=self.sQK_T) #TODO add inplace update
        
        out = vmap(lambda qk_t, v: torch.bmm(qk_t, v), in_dims=-2, out_dims=-2)(self.sQK_T, self.lV)
        out = torch.flatten(out, -2, -1)
        return self.linears["O"](out)

    
    @staticmethod
    def _softmax_jacobian(softmax_out: torch.Tensor): #softmax_out.shape -> N | 1xN
        """
        Returns the Jacobian of the softmax within the multihead attention

        Args:
            softmax_out (torch.Tensor): The output of the softmax within the multihead attention

        Returns:
            torch.Tensor: The Jacobian of the softmax 
        """
        softmax_out = torch.squeeze(softmax_out)
        n = softmax_out.shape[-1]
        
        jac_base = -softmax_out.view(n, 1) * softmax_out.view(1, n)
        diag = softmax_out*(1-softmax_out)
        jac_base[torch.arange(n), torch.arange(n)] = diag
        
        return jac_base

    @cleanup("lV", "lK", "lQ", "sQK_T")
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
        
        dL_dsQKT = vmap(lambda dl_dout, v: torch.bmm(dl_dout, torch.transpose(v, -1,-2)), in_dims=-2, out_dims=-2)(dL_dAtt, self.lV)
        #self.lV = None
        
        # vmap across 3 dims BxCxH 
        J_sQKT = vmap(vmap(vmap(self._softmax_jacobian)))(self.sQK_T)  # sQK_T.shape -> BxCxHxC 
        dL_dQKT = torch.squeeze(vmap(vmap(vmap(torch.mm)))(dL_dsQKT.unsqueeze(-2), J_sQKT)) * self.inv_sqrt_d
        
        # TODO verifiy this section
        dL_dlQ = vmap(lambda dl_dqkt, k: torch.bmm(dl_dqkt, k), in_dims=-2, out_dims=-2)(dL_dQKT, self.lK)  # k.T not necessary as its k.T.T
        #self.lK = None
        dL_dlKT = vmap(lambda dl_dqkt, q: torch.bmm(torch.transpose(q, -1,-2), dl_dqkt), in_dims=-2, out_dims=-2)(dL_dQKT, self.lQ)  
        #self.lQ = None
        dL_dlV = vmap(lambda dl_datt, sqkt: torch.bmm(torch.transpose(sqkt, -2, -1), dl_datt), in_dims=-2, out_dims=-2)(dL_dAtt, self.sQK_T) 
        #self.sQK_T = None
        
        dL_dQ = self.linears["Q"].backward_p1(self.dropouts["Q"].backward_p1(torch.flatten(dL_dlQ, -2, -1)))
        dL_dK = self.linears["K"].backward_p1(self.dropouts["K"].backward_p1(torch.flatten(torch.vmap(lambda dl_dlkt: torch.transpose(dl_dlkt, -1, -2), in_dims=-2, out_dims=-2)(dL_dlKT), -2, -1)))
        dL_dV = self.linears["V"].backward_p1(self.dropouts["V"].backward_p1(torch.flatten(dL_dlV, -2, -1)))
        
        return dL_dQ, dL_dK, dL_dV
        
        
    def backward_p2(self): 
        """
        Calculates the gradients of the parameters in the linear layers
        """
        #TODO add sychronize streams within model
        if self.device == "cuda":
            for k, s in zip(self.linears.keys(), self.streams):
                with torch.cuda.stream(s):
                    self.linears[k].backward_p2()
        else:
            for k in self.linears.keys():
                self.linears[k].backward_p2()          


#TODO will have to update for 3D parallelism (DP)
class BatchNorm2D(Layer):
    def __init__(self, in_channels: int, eps: float=1e-05, momentum =0.1):
        """
        Initializes a BatchNorm2D layer

        Args:
            in_channels (int): the number of input channels of the forward input 
            eps (float, optional): eps used to prevent div by 0. Defaults to 1e-05.
            momentum (float, optional): the value used for the running_mean and running_var computation. Defaults to 0.1.
        
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
        self.eps = eps
        self.momentum = momentum
        self.in_channels = in_channels
        
        self.gamma = torch.ones(in_channels)
        self.beta = torch.zeros(in_channels)
        
        self.gamma_g = torch.zeros(in_channels)
        self.beta_g = torch.zeros(in_channels)
        
        self.r_mean = torch.zeros(1, in_channels, 1, 1)
        self.r_var = torch.ones(1, in_channels, 1, 1)
        
        self.training = True
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the batchNorm2D

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: The output of the BAtchNorm2D Layer
        """
        if self.training:
            mean = x.mean(dim=[0,2,3], keepdim=True)
            var = ((x-mean)**2).mean(dim=[0,2,3], keepdim=True)
            self.r_mean = (1 - self.momentum) * self.r_mean + self.momentum * mean
            self.r_var = (1 - self.momentum) * self.r_var + self.momentum * var
        else:
            mean, var = self.r_mean, self.r_var
        
        self.x_sub_mean = x - mean
        self.var = var + self.eps
        self.norm_x = (self.x_sub_mean)/torch.sqrt(self.var)
        out = self.gamma.view(1, self.in_channels, 1, 1)*self.norm_x + self.beta.view(1, self.in_channels, 1, 1)
        return out
    
    @cleanup("var", "x_sub_mean")
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the BatchNorm2D output and returns the derivative of the loss with respect to the BatchNorm2D input

        Args:
            dL_dout (torch.Tensor): The derivative of the loss with respect to the BatchNorm2D output

        Returns:
            torch.Tensor: The derivatives of the loss with respect to the BatchNorm2D inputs
        """
        in_shape = dL_dout.shape
        self.dL_dout = dL_dout
        # https://stackoverflow.com/questions/67968913/derivative-of-batchnorm2d-in-pytorch
        
        B = in_shape[0]*in_shape[2]*in_shape[3]
        
        dL_dxnorm = self.dL_dout * self.gamma.view(1, -1, 1, 1)
        dL_dvar = (-0.5 * dL_dxnorm * (self.x_sub_mean)).sum((0, 2, 3), keepdim=True)  * ((self.var) ** -1.5)
        dL_dmean = (-1.0 / torch.sqrt(self.var) * dL_dxnorm).sum((0, 2, 3), keepdim=True) + (dL_dvar * (-2.0 * (self.x_sub_mean)).sum((0, 2, 3), keepdim=True) / B)
        return (dL_dxnorm / torch.sqrt(self.var)) + (2.0 * dL_dvar * (self.x_sub_mean) / B) + (dL_dmean / B) 
    
        
    @cleanup("dL_dout", "norm_x")
    def backward_p2(self):
        """
        Calculates the gradients of the parameters in the BatchNorm2D
        """
        self.gamma_g[:] = torch.sum(self.dL_dout*self.norm_x, dim=[0,2,3])
        self.beta_g[:] = torch.sum(self.dL_dout, dim=[0,2,3])
        
      
class NLPLayerNorm(Layer):
    def __init__(self, dim: int, dim_size: int, eps:Optional[float]=1e-08):
        """
        Initializes the NLP Layer Norm

        Args:
            dim (int): Embedding dim
            dim_size (int): Size of embedding dim
            eps (Optional[float]): eps used to prevent div by 0. Defaults to 1e-08.
        
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
        self.dim = dim #seqdim for nlp
        self.dim_size = dim_size
        self.eps = eps
        
        self.gamma = torch.ones(dim_size)
        self.bias = torch.zeros(dim_size)
        
        self.gamma_g = torch.zeros(dim_size)
        self.bias_g = torch.zeros(dim_size)
        
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the NLP Layer Norm

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: The output of the NLP layer norm
        """

        mean = torch.mean(x, dim=self.dim, keepdim=True)
        self.x_sub_mean = x-mean
        self.var = torch.mean(torch.square(self.x_sub_mean), dim=self.dim, keepdim=True) + self.eps
        self.norm_x = (self.x_sub_mean)/torch.sqrt(self.var)
        return self.norm_x*self.gamma.view(1,1,-1) + self.bias.view(1,1,-1)
    
    def _norm_jacobian(self):
        """Returns the Jacobian of the NLP layer norm

        Returns:
            torch.Tensor: The Jacobian of the NLP layer norm
        """
        # F_Jij is pass vectors x,z and scalar v of vector x
        def _F_Jij(x, z, v):
            const_n2 = self.dim_size
            f = lambda __x, __z, _v, g: g*((-torch.sqrt(_v)/self.dim_size)-__z*((__x)/const_n2))/_v
            def i_for_j(_x, _z):
                return vmap(f, in_dims=(None, 0, None, 0))(_x, _z, v, self.gamma)
            return vmap(i_for_j, in_dims=(0,None))(x, z)

        def _F_Jii(x, z, v):
            const_n2 = self.dim_size
            f = lambda __x, __z, _v, g: g*(((1-1/self.dim_size)*torch.sqrt(_v))-__z*((__x)/const_n2))/_v
            return vmap(f, in_dims=(0,0,None,0))(x, z, v, self.gamma)

        def _diag_set(jac, _diag):
            jac.diagonal()[:]= _diag
            return jac
        
        jac_base = vmap(vmap(_F_Jij))(self.x_sub_mean, self.norm_x,  self.var).squeeze()
        diag = vmap(vmap(_F_Jii))(self.x_sub_mean, self.norm_x, self.var).squeeze()
        return vmap(vmap(_diag_set))(jac_base, diag)
        
    @cleanup("x_sub_mean", "var")
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the NLP layer norm output and returns the derivative of the loss with respect to the NLP layer norm input

        Args:
            dL_dout (torch.Tensor): The derivative of the loss with respect to the NLP layer norm output

        Returns:
            torch.Tensor: The derivatives of the loss with respect to the NLP layer norm inputs
        """
        self.dL_dout = dL_dout
        
        dx_hat = dL_dout * self.gamma.view(1,1,-1)  
        dL_dvar = torch.sum(dx_hat * self.x_sub_mean, dim=-1, keepdim=True) * -.5 * self.var**(-1.5)
        dL_dmean = torch.sum(dx_hat/-torch.sqrt(self.var), dim=-1, keepdim=True) + dL_dvar * torch.mean(-2. * self.x_sub_mean, dim=-1, keepdim=True)
        return dx_hat / torch.sqrt(self.var) + dL_dvar * 2. * self.x_sub_mean /self.dim_size + dL_dmean / self.dim_size
        
        #J = self._norm_jacobian()
        #return vmap(vmap(torch.mm))(dL_dout.unsqueeze(-2), J).squeeze()

    @cleanup("dL_dout", "norm_x")
    def backward_p2(self):
        """
        Computes the gradients of the parameter in the NLP layer norm
        """
        self.bias_g[:] = torch.sum(self.dL_dout, dim=tuple(range(self.dL_dout.ndim)[:-1]))
        self.gamma_g[:] = torch.sum(self.dL_dout*self.norm_x, dim=tuple(range(self.dL_dout.ndim)[:-1]))
        self.dL_dout = None
        self.norm_x = None
      
        
class NLPRMSNorm(Layer):
    def __init__(self, dim: int, dim_size: int, eps: float= 1e-08):
        """
        Initializes the NLP RMS Norm

        Args:
            dim (int): Embedding dim
            dim_size (int): Size of embedding dim
            eps (Optional[float]): eps used to prevent div by 0. Defaults to 1e-08.
        
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
        self.dim = dim #seqdim for nlp
        self.dim_size = dim_size
        self.eps = eps
        
        self.weights = torch.ones(dim_size)
        
        self.weights_g = torch.zeros(dim_size)
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through the NLP RMS norm

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: The output of the NLP RMS norm
        """
        self.inputs = x
        self.mean_pow2 = torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        self.rms_norm_x = x*torch.rsqrt(self.mean_pow2)
        return self.rms_norm_x*self.weights
    
    @cleanup("inputs", "mean_pow2")
    def _rmsnorm_jacobian(self):
        """Returns the Jacobian of the NLP RMS norm

        Returns:
            torch.Tensor: The Jacobian of the NLP RMS norm
        """
        # F_Jij is passed vectors x,z and scalar v of vector x
        def _F_Jij(x,z, mp2):
            f = lambda __x, __z, _mp2, w: w*((-__x*(1/self.dim_size)*__z)/_mp2)
            def i_for_j(_x, _z):
                return vmap(f, in_dims=(None, 0, None, 0))(_x, _z, mp2, self.weights)
            return vmap(i_for_j, in_dims=(0,None))(x, z)

        def _F_Jii(x, z, mp2):
            f = lambda __x, __z, _mp2, w: w*((torch.sqrt(_mp2)-__x*(1/self.dim_size)*__z)/_mp2)
            return vmap(f, in_dims=(0,0,None,0))(x,z,mp2,self.weights)


        def _diag_set(jac, _diag):
            jac.diagonal()[:]= _diag
            return jac
        
        jac_base = vmap(vmap(_F_Jij))(self.inputs, self.rms_norm_x,  self.mean_pow2).squeeze()
        diag = vmap(vmap(_F_Jii))(self.inputs, self.rms_norm_x, self.mean_pow2).squeeze()
        return vmap(vmap(_diag_set))(jac_base, diag)

    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the NLP RMS norm output and returns the derivative of the loss with respect to the NLP RMS norm input

        Args:
            dL_dout (torch.Tensor): The derivative of the loss with respect to the NLP RMS norm output

        Returns:
            torch.Tensor: The derivatives of the loss with respect to the NLP RMS norm inputs
        """
        self.dL_dout = dL_dout
        J = self._rmsnorm_jacobian()
        return vmap(vmap(torch.mm))(dL_dout.unsqueeze(-2), J).squeeze()
    
    @cleanup("dL_dout", "rms_norm_x")
    def backward_p2(self):
        """
        Computes the gradients of the parameter in the NLP layer norm
        """
        self.weights_g[:] = torch.sum(self.dL_dout*self.rms_norm_x, dim=tuple(range(self.dL_dout.ndim)[:-1]))
        self.dL_dout = None
        self.rms_norm_x = None

        
class BertBlock(Layer):
    def __init__(self, emb_dim: int, num_heads: int, dim_ff:int, activation:Layer=ReLU, eps:float=1e-08, p:float=0.1, device = "cuda"):
        """
        Initialize BERT block

        Args:
            emb_dim (int): size of the embedding dim
            num_heads (int): number of heads in the multihead attention
            dim_ff (int): size of hidden dim in the ffn
            activation (Layer, optional): activation function used on hidden layer in ffn. Defaults to ReLU.
            eps (float, optional): the eps used in the layer norms. Defaults to 1e-08.
            p (float, optional): the probability used in the dropouts. Defaults to 0.1.
        
        Attributes:
            multihead (MultiHeadAttention): multihead attention layer
            linears (Dict[Dense]): linear layers
            ff_act (Layer): activation function used on hidden layer in ffn
            norms (Dict[NLPLayerNorm]): layer norms
            dropouts (Dict[Dropout]): dropouts
            device (str): device used for computation
            streams (List[torch.cuda.Stream]): streams used for parallel computation of backward_p2
        """
        self.multihead = MultiHeadAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.linears = {0: Dense(emb_dim, dim_ff),
                        1: Dense(dim_ff, emb_dim)}
        self.ff_act = activation()
        self.norms = {"multi_head": NLPRMSNorm(-1, emb_dim, eps=eps),
                      "ff": NLPRMSNorm(-1, emb_dim, eps=eps)}
        self.dropouts = {"multi_head": Dropout(p=p),
                         "ff": Dropout(p=p)}
        
        if device == "cuda":
            self.device = "cuda"
            self.streams = []
            for _ in range(5):  # 5 Layers with parameters
                self.streams.append(torch.cuda.Stream())
        else:
            self.device = "cpu"
        
    def forward(self, x:torch.Tensor):
        """
        Performs a forward pass through the BERT block. 

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor of Bert block 
        """
        mh_out = self.multihead(x, x, x) + x
        norm_mh_out = self.norms["multi_head"](mh_out)
        norm_mh_out = self.dropouts["multi_head"](norm_mh_out)
        
        ff1 = self.linears[0](norm_mh_out)
        a = self.ff_act(ff1)
        ff2 = self.linears[1](a) + norm_mh_out
        ff2_norm = self.norms["ff"](ff2)
        return self.dropouts["ff"](ff2_norm)
    
    def backward_p1(self, dL_dout:torch.Tensor):
        """
        Takes the derivative of the loss with respect to the BERT block output and returns the derivative of the loss with respect to the BERT block input

        Args:
            dL_dout (torch.Tensor): Takes the derivative of the loss with respect to the BERT block output 

        Returns:
            torch.Tensor: The derivative of the loss with respect to the BERT block input

        """
        dL_dff2norm = self.dropouts["ff"].backward_p1(dL_dout)
        dL_dff2 = self.norms["ff"].backward_p1(dL_dff2norm)
        dL_da = self.linears[1].backward_p1(dL_dff2)
        dL_dff1 = self.ff_act.backward_p1(dL_da)
        dL_dnormmhout = self.linears[0].backward_p1(dL_dff1) + dL_dff2
        
        dL_dnormmhout = self.dropouts["multi_head"].backward_p1(dL_dnormmhout)
        dL_dmhout = self.norms["multi_head"].backward_p1(dL_dnormmhout)
        dL_din1 = torch.sum(torch.stack(self.multihead.backward_p1(dL_dmhout)),dim=0)
        return dL_din1 + dL_dmhout  # dLdmhout == dL_din2
    
    def backward_p2(self):
        """
        Computes the gradients of the parameter in the BERT block
        """
        if "cuda" in str(self.device):
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
            
class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, k_size: Union[Sequence[int], int], bias: bool=True,  padding: Union[bool, int]=False, stride: Union[Sequence[int], int]=1):
        """
        Initializes Conv2D layer

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            k_size (Union[Sequence[int], int]): size of kernel
            bias (bool): whether the conv2d layer has bias. Defaults to True.
            padding (Union[bool, int]): the padding either side of the input feature. Defaults to False.
            stride (Union[Sequence[int], int], optional): the stride pattern followed by the kernel. Defaults to 1.

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
        if isinstance(k_size, int):
            k_size = (k_size, k_size)
        self.kernel = torch.randn(out_channels, in_channels, *k_size)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.bias = None
        
        self.kernel_g = torch.zeros_like(self.kernel)
        if bias:
            self.bias_g = torch.zeros_like(self.bias)
        
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
        
    def forward(self, x:torch.Tensor):
        """
        Preforms a forward pass through the Conv2D layer

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor
        """
        self.inputs = x
        return F.conv2d(x, self.kernel, self.bias, stride=self.stride, padding=self.padding)
    
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the Conv2D Layer's output and returns the derivative of the loss with respect to the Conv2D Layer's input

        Args:
            dL_dout (torch.Tensor): derivative of the loss with respect to the Conv2D Layer's output

        Returns:
            torch.Tensor: derivative of the loss with respect to the Conv2D Layer's input
        """
        self.dL_dout = dL_dout
        #TODO fix bug when stride is greater than 3
        dL_din = F.conv_transpose2d(dL_dout, self.kernel, stride=self.stride, padding=self.padding, output_padding=(self.stride[0]-1,self.stride[1]-1))
        return dL_din
    
    @cleanup("dL_dout", "inputs")
    def backward_p2(self):
        """
        Computes the gradients of the parameter within the Conv2D layer
        """
        if isinstance(self.bias, torch.Tensor):
            self.bias_g[:] = torch.sum(self.dL_dout, dim=(0, 2, 3))
        
        #print(self.inputs.unsqueeze(-3).shape, self.dL_dout.unsqueeze(-3).unsqueeze(-3).shape)

        def _dL_dk_fn(feature, kernal, stride, padding):
            new_stride = [stride[0]*kernal.shape[2], stride[1]*kernal.shape[3]]
            out = F.conv2d(feature, kernal, stride=tuple(new_stride), padding=padding).squeeze(0)
            return out
        # witchcraft (apply a conv between each input and output channel and sum)
        self.kernel_g[:] = torch.sum(vmap(vmap(vmap(_dL_dk_fn, in_dims=(0, None)), in_dims=(None, 0)))(self.inputs.unsqueeze(-3), self.dL_dout.unsqueeze(-3).unsqueeze(-3), stride=self.stride, padding=self.padding),dim=0)


class MaxPool2D(Layer):
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
    
    def forward(self, x: torch.Tensor):
        """
        Preforms a forward pass through the MaxPool2D layer

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        out, self.indices = F.max_pool2d(x, kernel_size=self.k_size, stride=self.stride, padding=self.padding, return_indices=True)
        return out
    
    @cleanup("indices")
    def backward_p1(self, dL_dout: torch.Tensor):
        """
        Takes the derivative of the loss with respect to the MaxPool2D Layer's output and returns the derivative of the loss with respect to the MaxPool2D Layer's input

        Args:
            dL_dout (torch.Tensor): derivative of the loss with respect to the MaxPool2D Layer's output

        Returns:
            torch.Tensor: derivative of the loss with respect to the MaxPool2D Layer's input
        """
        out = F.max_unpool2d(dL_dout, self.indices, kernel_size=self.k_size, stride=self.stride, padding=self.padding)
        return out

    def backward_p2(self):
        pass


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
    
    def backward_p2(self):
        pass

class BasicResNetBlock(Layer):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, device: str="cuda"):
        """
        Initializes a Basic ResNet Block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): he stride pattern followed by the kernel of the first conv layer. Defaults to 1.
        
        Attributes:
            convs (List[Conv2D]): List of Conv2D layers
            batchnorms (List[BatchNorm2D]): List of BatchNorm2D layers)
            relus (List[ReLU]): List of ReLU layers])
            shortcut (List[Layer]): List of layers that are used to connect the input and output of the Basic ResNet Block
            streams (List[torch.cuda.Stream]): List of streams used to parallelize the execution of the Basic ResNet Block
            device (str): Device used to parallelize the execution of the Basic ResNet Block
        """
        self.convs = [Conv2D(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                      Conv2D(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
                      ]
        self.batchnorms = [BatchNorm2D(out_channels),
                           BatchNorm2D(out_channels)
                           ]
        self.relus = [ReLU(), ReLU()]
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.shortcut = [Conv2D(in_channels, out_channels*self.expansion, 1, stride=stride, bias=False),
                             BatchNorm2D(out_channels*self.expansion)
                             ]
        else:
            self.shortcut = []
        
        if device == "cuda":
            self.streams = []
            self.device = "cuda"
            for _ in range(len(self.convs) + len(self.batchnorms) + len(self.shortcut)):
                self.streams.append(torch.cuda.Stream())
        else:
            self.device = "cpu"
    
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
        
    def backward_p2(self):
        """
        calcualtes the gardients of the parameter within the BasicResNetBlock
        """
        if self.device == "cuda":
            for i, layer in enumerate(self.convs):
                with torch.cuda.stream(self.streams[i]):
                    layer.backward_p2()
            for i, layer in enumerate(self.batchnorms):
                with torch.cuda.stream(self.streams[i+len(self.convs)]):
                    layer.backward_p2()
            for i, layer in enumerate(self.shortcut):
                with torch.cuda.stream(self.streams[i+len(self.convs)+len(self.batchnorms)]):
                    layer.backward_p2()
        else:
            for i, layer in enumerate(self.convs):
                layer.backward_p2()
            for i, layer in enumerate(self.batchnorms):
                layer.backward_p2()
            for i, layer in enumerate(self.shortcut):
                layer.backward_p2()
            

class ResNetBottleneck(Layer):
    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, device: str="cuda"):
        """
        Initializes a Basic ResNet Block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): he stride pattern followed by the kernel of the first conv layer. Defaults to 1.
        
        Attributes:
            convs (List[Conv2D]): List of Conv2D layers
            batchnorms (List[BatchNorm2D]): List of BatchNorm2D layers)
            relus (List[ReLU]): List of ReLU layers])
            shortcut (List[Layer]): List of layers that are used to connect the input and output of the Basic ResNet Block
            streams (List[torch.cuda.Stream]): List of streams used to parallelize the execution of the Basic ResNet Block
            device (str): Device used to parallelize the execution of the Basic ResNet Block
        """
        self.convs = [Conv2D(in_channels, out_channels, 1, bias=False),
                      Conv2D(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                      Conv2D(out_channels, self.expansion*out_channels, 1, bias=False)
                      ]
        self.batchnorms = [BatchNorm2D(out_channels),
                           BatchNorm2D(out_channels),
                           BatchNorm2D(self.expansion*out_channels)
                           ]
        self.relus = [ReLU(), ReLU(), ReLU()]
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.shortcut = [Conv2D(in_channels, out_channels*self.expansion, 1, stride=stride, bias=False),
                             BatchNorm2D(out_channels*self.expansion)
                             ]
        else:
            self.shortcut = []
        
        if device == "cuda":
            self.streams = []
            self.device = "cuda"
            for _ in range(len(self.convs) + len(self.batchnorms) + len(self.shortcut)):
                self.streams.append(torch.cuda.Stream())
        else:
            self.device = "cpu"
            
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

    def backward_p2(self):
        """
        Calculaes the gradients of the parameters within the ResNetBottleneck
        """
        if self.device == "cuda":
            for i, layer in enumerate(self.convs):
                with torch.cuda.stream(self.streams[i]):
                    layer.backward_p2()
            for i, layer in enumerate(self.batchnorms):
                with torch.cuda.stream(self.streams[i+len(self.convs)]):
                    layer.backward_p2()
            for i, layer in enumerate(self.shortcut):
                with torch.cuda.stream(self.streams[i+len(self.convs)+len(self.batchnorms)]):
                    layer.backward_p2()
        
        elif self.device == "cpu":
            for layer in self.convs:
                layer.backward_p2()
            for layer in self.batchnorms:
                layer.backward_p2()
            for layer in self.shortcut:
                layer.backward_p2()
        
class ResNet(Layer):
    # For Reference https://github.com/henryqin1997/CIFAR10-ResNet50-PyTorch/blob/master/models/resnet.py
    # Needs to be parallised just for testing
    def __init__(self, block: Union[BasicResNetBlock , ResNetBottleneck], num_blocks: list, num_classes: int=10, device:str = "cuda"):
        self.in_planes = 64

        if device == "cuda":
            self.device = "cuda"
            for _ in range(3):  # conv1, bn1, dense
                self.streams.append(torch.cuda.Stream())
            for _ in range(4):  # layers
                self.streams.append(torch.cuda.Stream())
        else:
            self.device = "cpu"
        
        self.conv1 = Conv2D(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2D(64)
        
        self.layers1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layers2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layers3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layers4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = Dense(512*block.expansion, num_classes)
        
        self.relu = ReLU()
        self.avgpool = AvgPool2D(4)
        
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, device=self.device))
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
        self.preflattened = out.shape
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def backward_p1(self, dL_dout):
        dL_davgpool = self.linear.backward_p1(dL_dout)
        dL_davgpool = dL_davgpool.view(*self.preflattened)
        dL_dl4 = self.avgpool.backward_p1(dL_davgpool)
        dL_dl3 = self.layers4.backward_p1(dL_dl4)
        dL_dl2 = self.layers3.backward_p1(dL_dl3)
        dL_dl1 = self.layers2.backward_p1(dL_dl2)
        dL_drelu = self.layers1.backward_p1(dL_dl1)
        dL_dbn = self.relu.backward_p1(dL_drelu)
        dL_dconv = self.bn1.backward_p1(dL_dbn)
        return self.conv1.backward_p1(dL_dconv)
    
    def backward_p2(self):
        if self.device == "cuda":
            with self.streams[0]:
                self.conv1.backward_p2()
            with self.streams[1]:
                self.bn1.backward_p2()
            with self.streams[2]:
                self.linear.backward_p2()
            with self.streams[3]:
                self.layers1.backward_p2()
            with self.streams[4]:
                self.layers2.backward_p2()
            with self.streams[5]:
                self.layers3.backward_p2()
            with self.streams[6]:
                self.layers4.backward_p2()
        else:
            self.conv1.backward_p2()
            self.bn1.backward_p2()
            self.linear.backward_p2()
            self.layers1.backward_p2()
            self.layers2.backward_p2()
            self.layers3.backward_p2()
            self.layers4.backward_p2()


class SiLU(Activation):
    def forward(self, x: torch.Tensor):
        self.inputs = x
        return x * torch.sigmoid(x)

    @cleanup("inputs")
    def backward_p1(self, dL_dout: torch.Tensor):
        e_x = torch.exp(-self.inputs)
        out = dL_dout * ((1+e_x) + self.inputs*e_x)/((1+e_x)**2)

    def backward_p2(self):
        pass


class llamaFF(Layer):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier = None, device:str = "cuda"):
        hidden_dim = int(2 * hidden_dim / 3)
        
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.linears = [
            Dense(dim, hidden_dim),
            Dense(hidden_dim, dim),
            Dense(dim, hidden_dim)
        ]
        
        self.silu = SiLU()

        self.streams = []
        if device == "cuda":
            self.device = "cuda"
            for _ in range(3):  # 3 dense
                self.streams.append(torch.cuda.Stream())
        else:
            self.device = "cpu"
    
    def forward(self, x: torch.Tensor):
        l0 = self.linears[0](x)
        l2 = self.linears[2](x)
        self.l2 = l2
        silu_out = self.silu.initial_pass(l0)
        self.silu_out = silu_out
        return self.linears[1](self.silu(l0)*l2)
    
    @cleanup("l2", "silu_out")
    def backward_p1(self, dL_dout:torch.Tensor):
        dL_dl2in = self.linears[1].backward_p1(dL_dout)
        dL_dl0 = self.silu.backward_p1(dL_dl2in)
        dL_dx1 = self.linears[0].backward_p1(dL_dl0) * self.l2
        dL_dx2 = self.linears[2].backward_p1(dL_dl2in) * self.silu_out
        return dL_dx1 + dL_dx2
    
    def backward_p2(self):
        if self.device == "cuda":  
            for l, s in zip(self.linears, self.streams):
                with s:
                    l.backward_p2()
        else:
            for l in self.linears:
                l.backward_p2()

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class RotaryEmbeddings(Layer):
    def __init__(self, freq_cis):
        ...

class GroupedMultiQueryAttention(Layer):
    def __init__(self, emb_dim: int, num_heads: int, num_kv_heads:int, max_seq_len:int):
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = emb_dim // num_heads
        self.inv_square_d = 1/torch.sqrt(torch.tensor(self.head_dim))
        
        self.linears = {"Q": Dense(self.emb_dim, self.num_heads*self.head_dim, bias=False),
                        "K": Dense(self.emb_dim, self.num_kv_heads*self.head_dim, bias=False),
                        "V": Dense(self.emb_dim, self.num_kv_heads*self.head_dim, bias=False),
                        "O": Dense(self.num_heads*self.head_dim, self.emb_dim, bias=False)
                        }

    def initial_pass(self, x: torch.Tensor):
        batch_size, seqlen, _ = x.shape
        self.device = x.device

        self.k_cache = torch.zeros(batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim).to(x.device)
        self.v_cache = torch.zeros(batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim).to(x.device)
        
        xq, xk, xv = self.linears["Q"].initial_pass(x), self.linears["K"].initial_pass(x), self.linears["V"].initial_pass(x)

        self.linears["O"].initial_pass(x)

    
    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seqlen, _ = x.shape

        xq, xk, xv = self.linears["Q"](x), self.linears["K"](x), self.linears["V"](x)

        xq = xq.view(batch_size, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen, self.num_kv_heads, self.head_dim)
        self.xv = xv.view(batch_size, seqlen, self.num_kv_heads, self.head_dim)

        xq, self.xk = apply_rotary_emb(xq, xk, freq_cis)
        self.xq = torch.cat(xq.unsqueeze(-3).chunk(self.n_rep, dim=-2), dim=-3)
        #self.xq = xq #torch.chunk(xq, self.n_rep, dim=-2)

        print(xk.shape, xq.shape)
        QK_T = vmap(lambda q, k: vmap(lambda _q: torch.bmm(_q, k.transpose(-1,-2)), in_dims=-2, out_dims=-2)(q), in_dims=-2, out_dims=-2)(self.xq, self.xk) * self.inv_square_d
        QK_T = torch.flatten(QK_T, -3, -2)
        #vmap(lambda k: vamp(torch.bmm(q, k.transpose(-1,-2)))(self.xk)    * self.inv_square_d
        print(QK_T.shape)
        #Q_KT = vmap(lambda q, k: torch.bmm(q, k.transpose(-1,-2)), in_dims=-2, out_dims=-2)(self.xq, self.xk) * self.inv_square_d
        if mask is not None:
            QK_T = vmap(lambda x: x + mask, in_dims=-2, out_dims=-2)(QK_T)
        def _softmax(x):
            e_x = torch.exp(x)
            return e_x / torch.sum(e_x)
        self.sQK_T = vmap(vmap(vmap(_softmax)))(QK_T)
        self.sQK_T = torch.cat(self.sQK_T.unsqueeze(-3).chunk(self.n_rep, dim=-2), dim=-3)
        print(self.sQK_T.shape)
        out = vmap(lambda sqkt, v: vmap(lambda _sqkt: torch.bmm(_sqkt, v), in_dims=-2, out_dims=-2)(sqkt), in_dims=-2, out_dims=-2)(self.sQK_T, self.xv)
        out = torch.flatten(out, -3, -2)
        out = torch.flatten(out, -2, -1)
        return self.linears["O"](out)

    @staticmethod
    def _softmax_jacobian(softmax_out: torch.Tensor): #softmax_out.shape -> N | 1xN
        """
        Returns the Jacobian of the softmax within the multihead attention

        Args:
            softmax_out (torch.Tensor): The output of the softmax within the multihead attention

        Returns:
            torch.Tensor: The Jacobian of the softmax 
        """
        softmax_out = torch.squeeze(softmax_out)
        n = softmax_out.shape[-1]
        
        jac_base = -softmax_out.view(n, 1) * softmax_out.view(1, n)
        diag = softmax_out*(1-softmax_out)
        jac_base[torch.arange(n), torch.arange(n)] = diag
        
        return jac_base

    def backward_p1(self, dL_dout: torch.Tensor):
        dL_dAtt = self.linears["O"].backward_p1(dL_dout)
        dL_dAtt = torch.cat(dL_dAtt.unsqueeze(-2).chunk(self.num_heads, dim=-1), dim=-2)

        dL_dsQKT = vmap(lambda dl_dout, v: torch.bmm(dl_dout, torch.transpose(v, -1,-2)), in_dims=-2, out_dims=-2)(dL_dAtt, self.values)

        J_sQKT = vmap(vmap(vmap(self._softmax_jacobian)))(self.sQK_T)  # sQK_T.shape -> BxCxHxC 
        dL_dQKT = torch.squeeze(vmap(vmap(vmap(torch.mm)))(dL_dsQKT.unsqueeze(-2), J_sQKT)) * self.inv_sqrt_d

        dL_dlQ = vmap(lambda dl_dqkt, k: torch.bmm(dl_dqkt, k), in_dims=-2, out_dims=-2)(dL_dQKT, self.values)  # k.T not necessary as its k.T.T  
        dL_dlKT = vmap(lambda dl_dqkt, q: torch.bmm(torch.transpose(q, -1,-2), dl_dqkt), in_dims=-2, out_dims=-2)(dL_dQKT, self.xq)  
        dL_dlV = vmap(lambda dl_datt, sqkt: torch.bmm(torch.transpose(sqkt, -2, -1), dl_datt), in_dims=-2, out_dims=-2)(dL_dAtt, self.sQK_T) 

        dL_dQ = self.linears["Q"].backward_p1(torch.flatten(dL_dlQ, -2, -1))
        dL_dK = self.linears["K"].backward_p1(torch.flatten(torch.vmap(lambda dl_dlkt: torch.transpose(dl_dlkt, -1, -2), in_dims=-2, out_dims=-2)(dL_dlKT), -2, -1))
        dL_dV = self.linears["V"].backward_p1(torch.flatten(dL_dlV, -2, -1))

        return dL_dQ + dL_dK + dL_dV
    
    def backward_p2(self):
        if self.device == "cuda":
            for k, s in zip(self.linears.keys(), self.streams):
                with torch.cuda.stream(s):
                    self.linears[k].backward_p2()
        else:
            for k in self.linears.keys():
                self.linears[k].backward_p2()  


if __name__ == "__main__":

    def test(layer, x, dL_dout):
        with torch.no_grad():
            layer.forward(x)
            my_back = layer.backward_p1(dL_dout)
            layer.backward_p2()
        true_back = torch.autograd.functional.vjp(layer.forward, x, dL_dout)[1]
        print(my_back.shape, true_back.shape)
        if len(x.shape) == 3:
            print(my_back[:2,:2,:2])
            print(true_back[:2,:2,:2])
        elif len(x.shape) == 2:
            print(my_back[:2,:2])
            print(true_back[:2,:2])
        else:
            print(my_back[:2,:2,:2, :2])
            print(true_back[:2,:2,:2, :2])
        print(my_back.mean())
        num_correct = (torch.isclose(my_back, true_back, rtol=0.01, atol=0.00001)).sum()
        print(f"Num correct: {num_correct}/{my_back.numel()}")
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
        layer = GroupedMultiQueryAttention(80, 8, 4, 30)
        freqs_cis = precompute_freqs_cis(80 // 8, 30)
        print(freqs_cis.shape)
        with torch.no_grad():
            layer.forward(x, freq_cis=freqs_cis)
            exit()
            my_back = layer.backward_p1(dL_dout)
            layer.backward_p2()
        true_back = torch.autograd.functional.vjp(lambda _x: layer.forward(_x, _x, _x), x, dL_dout)[1]
        print(my_back[0,0])
        print(true_back[0,0])
        print(torch.allclose(my_back, true_back))
    
    def test_layernorm():
        m = torch.distributions.Uniform(1, 3)
        test(NLPLayerNorm(-1, 80), m.sample((16,24,80)), torch.ones(16,24,80))
    
    def test_bert():
        m = torch.distributions.Uniform(1, 3)
        x = torch.randn(16, 24, 80)
        x = m.sample(x.shape)
        dL_dout = torch.ones(16, 24, 80)
        layer = BertBlock(80, 8, 160, p=0)
        test(layer, x, dL_dout)
    
    def test_conv2D():
        image = torch.randn(16, 3, 80, 80)
        conv = Conv2D(3, 16, 3)
        test(conv, image, torch.ones(16,16,78,78))
    
    def test_batchnorm():
        m = torch.distributions.Uniform(1, 3)
        image = m.sample((16, 3, 80, 80))
        test(BatchNorm2D(3), image, torch.ones(16,3,80,80))

    def test_dense():
        x = torch.randn(16, 80)
        test(Dense(80, 160), x, torch.ones(16,160))
    
    def test_resnet():
        image = torch.randn(16, 3, 32, 32)
        resnet50 = ResNet(ResNetBottleneck, [3, 4, 6, 3], device="cpu")
        test(resnet50, image, torch.ones_like(resnet50(image)))
    
    def test_rmsnorm():
        m = torch.distributions.Uniform(1, 3)
        x = m.sample((16,24,80))
        dL_dout = torch.ones(16,24,80)
        layer = NLPRMSNorm(-1,80)
        test(layer, x, dL_dout)


    test_resnet()
