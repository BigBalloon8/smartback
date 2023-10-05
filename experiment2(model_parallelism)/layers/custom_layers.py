from typing import Any
import torch
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_size, output_size, batch_size):
        self.i_size = input_size
        self.o_size = output_size
        self.b_size = batch_size
        self.training = True
        
        self.params = {}
        self.grads = {}
        
        #Whether the derivative is independent of the input
        self.ind_der = False
        
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
    

class Dense(Layer):
    def __init__(self, input_size, output_size, batch_size):
        super().__init__(input_size, output_size, batch_size)
        self.params = {
            "w": torch.randn((self.i_size, self.o_size))*0.1,
            "b": torch.zeros((self.o_size))
            }

        self.grads = {
            "w": torch.zeros((self.i_size, self.o_size)),
            "b": torch.zeros((self.o_size))
            }
        
        self.dL_dout = torch.zeros((self.b_size, self.o_size))
        
        self.inputs = torch.zeros((self.b_size, self.i_size))
        self.out = torch.zeros((self.b_size, self.o_size))
        
        #jacobian of dense is constant for a batch
        self.ind_der = True
    
    def forward(self, x):
        self.inputs[:] = x 
        x = torch.mm(x, self.params["w"])
        torch.add(x, self.params["b"], out=self.out)
        return self.out
    
    def backward_p1(self, dL_dout):
        self.dL_dout[:] = dL_dout
        return torch.mm(dL_dout, self.get_jac())
    
    def backward_p2(self):
        self.grads["b"][:] = torch.mean(self.dL_dout, dim=0)
        self.grads["w"][:] = torch.mean(
            torch.einsum('bij,bjk->bik', 
                         self.inputs.view(self.b_size, self.i_size, 1), 
                         self.dL_dout.view(self.b_size, 1, 
                                    self.o_size)),
            dim=0
        )
        
    def get_jac(self):
        return self.params["w"].T
    
class Relu(Layer):
    def __init__(self, input_size, batch_size):
        super().__init__(input_size, input_size, batch_size)
        self.inputs = torch.zeros((self.b_size, self.i_size))
        self.out = torch.zeros((self.b_size, self.o_size))

    def forward(self, x):
        self.inputs[:] = x
        torch.maximum(x, 0, out=self.out)
        return self.out
    
    def backward_p1(self, dL_dout):
        dout_din = torch.where(self.inputs>0, 1.0, 0.0)
        return torch.bmm(dL_dout, dout_din)

    def backward_p2(self):
        pass


class BatchNorm(Layer):
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

        
        X_mu = self.inputs-self.b_mu
        var_sqrt_inv = 1./torch.sqrt(self.b_var + self.eps)
        
        dout_dXnorm = dL_dout * self.params["gamma"]
        
        return (1./self.b_size) * var_sqrt_inv * (self.b_size*dout_dXnorm - torch.sum(dout_dXnorm, dim=0) - self.x_norm*torch.sum(dout_dXnorm*self.x_norm, dim=0))

    
    def backward_p2(self):
        self.grads["beta"][:] = torch.sum(self.dL_dout, dim=0)
        self.grads["gamma"][:] = torch.sum(self.x_norm*self.dL_dout, dim=0)
        
class Conv2D(Layer):
    def __init__(self, in_channels, out_channels,  batch_size, kernal_size, stride=(1,1), padding=0):
        super().__init__(in_channels, out_channels, batch_size)
        self.padding = padding
        self.stride = stride
        self.params = {
            "k": torch.randn((out_channels, in_channels, *kernal_size)),
            "b": torch.randn((out_channels))
            }
        self.grads = {
            "k": torch.zeros((out_channels, in_channels, *kernal_size)),
            "b": torch.zeros((out_channels))
            
        }
    
    def forward(self, x):
        # may have to deal with initial input having no channel dim
        conv_kwargs = {"stride": self.stride, "padding": self.padding, "bias": self.params["b"]}
        if hasattr(self, "inputs"):
            self.inputs[:] = x
        else:
            self.inputs = x
        return torch.nn.functional.conv2d(self.inputs, self.params["k"], **conv_kwargs)
    
    def backward_p1(self, dL_dout):
        if hasattr(self, "dL_dout"):
            self.dL_dout[:] = dL_dout
        else:
            self.dL_dout = dL_dout

        conv_kwargs = {"stride": self.stride, "padding": self.padding}#, "groups":self.o_size}
        return torch.nn.functional.conv_transpose2d(dL_dout, self.params["k"], **conv_kwargs)

    def backward_p2(self):
        self.grads["b"] = torch.mean(torch.sum(self.dL_dout, dim=(-1,-2)), dim=0)
        #self.grads["k"] = torch.mean(torch.nn.functional.conv2d(self.inputs, self.dL_dout))
        
        for i in range(self.params["k"].size(0)):
            for j in range(self.params["k"].size(1)):
                #print(self.inputs[:,j].shape, self.dL_dout[:,i].shape)
                #torch.vmap(torch.nn.functional.conv2d)(self.inputs[:,j].unsqueeze(1).unsqueeze(1), self.dL_dout[:,i].unsqueeze(1).unsqueeze(1))
                #torch.nn.functional.conv2d(self.inputs[:,j].unsqueeze(1), self.dL_dout[:,i].unsqueeze(1).unsqueeze(1)[0]) 
                self.grads["k"][i, j] = torch.sum(torch.vmap(torch.nn.functional.conv2d)(self.inputs[:,j].unsqueeze(1).unsqueeze(1), self.dL_dout[:,i].unsqueeze(1).unsqueeze(1)), dim=0)
                #torch.sum(torch.nn.functional.conv2d(self.inputs[:, j].unsqueeze(1), self.dL_dout[:,i].unsqueeze(1), padding="same"), dim=0)

class Flatten:
    def __init__(self):
        self.in_shape = None
        
    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
    
    def forward(self, x, start_dim=1, end_dim=-1):
        self.in_shape = x.shape
        return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)
    
    def backward_p1(self, dL_dout):
        return torch.reshape(dL_dout, self.in_shape)
    
    def backward_p2(self):
        pass

class MaxPool2D(Layer):
    def __init__(self, batch_size, kernal_size=(2,2), stride=(2,2)):
        super().__init__(None, None, batch_size)
        self.kernal_size = kernal_size
        self.stride = stride
    
    def forward(self, x):
        if hasattr(self, "out"):
            self.out[:], self.indices[:] = torch.nn.functional.max_pool2d(x, self.kernal_size, self.stride, return_indices=True)
        else:
            self.out, self.indices = torch.nn.functional.max_pool2d(x, self.kernal_size, self.stride, return_indices=True)
        return self.out
        
    def backward_p1(self, dL_dout):
        return torch.nn.functional.max_unpool2d(dL_dout, self.indices,  self.kernal_size, self.stride)
    
    def backward_p2(self):
        pass
            
        

if __name__ == "__main__":
    input_channels = torch.randn(16, 3, 5, 5)
    dL_dout = torch.randn(16, 2, 3, 3)
    conv_layer = Conv2D(3, 2, 16, (3,3))
    conv_layer(input_channels)
    print(conv_layer.backward_p1(dL_dout).shape)
    conv_layer.backward_p2()
    print(conv_layer.grads["k"].shape, conv_layer.params["k"].shape)
    