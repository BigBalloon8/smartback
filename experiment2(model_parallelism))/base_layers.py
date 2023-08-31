import torch
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_size, output_size, batch_size):
        self.i_size = input_size
        self.o_size = output_size
        self.b_size = batch_size
        
        self.params = {}
        
        self.grads = {}
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def backward(self, dL_dout):
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
        
        self.inputs = torch.zeros((self.b_size, self.i_size))
        self.out = torch.zeros((self.b_size, self.o_size))
    
    def forward(self, x):
        self.inputs[:] = x 
        x = torch.mm(x, self.params["w"])
        torch.add(x, self.params["b"], out=self.out)
        return self.out
    
    def backward(self, dL_dout):
        self.grads["b"][:] = torch.mean(dL_dout, axis=0)
        self.grads["w"][:] = torch.mean(
            torch.einsum('bij,bjk->bik', 
                         self.inputs.view(self.b_size, self.i_size, 1), 
                         dL_dout.view(self.b_size, 1, 
                                    self.o_size)),
            axis=0
        )
        return torch.mm(dL_dout, self.params["w"].T)

class Relu(Layer):
    def __init__(self, input_size, output_size, batch_size):
        super().__init__(input_size, output_size, batch_size)
        self.inputs = torch.zeros((self.b_size, self.i_size))
        self.out = torch.zeros((self.b_size, self.o_size))
    
    def forward(self, x):
        self.inputs[:] = x
        torch.maximum(x, 0, out=self.out)
        return self.out
    
    def backward(self, dL_dout):
        dout_din = torch.where(self.inputs>0, 1.0, 0.0)
        return torch.bmm(dL_dout, dout_din)

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
    
    def backward(self, dL_dout):
        self.grads["beta"][:] = torch.sum(dL_dout, dim=0)
        self.grads["gamma"][:] = torch.sum(self.x_norm*dL_dout, dim=0)
        
        X_mu = self.inputs-self.b_mu
        var_sqrt_inv = 1./torch.sqrt(self.b_var + self.eps)
        
        dout_dXnorm = dL_dout * self.params["gamma"]
        
        return (1./self.b_size) * var_sqrt_inv * (self.b_size*dout_dXnorm - torch.sum(dout_dXnorm, dim=0) - self.x_norm*torch.sum(dout_dXnorm*self.x_norm, dim=0))
        