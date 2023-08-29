import torch
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_size, output_size, batch_size, params=True):
        self.i_size = input_size
        self.o_size = output_size
        self.b_size = batch_size
        self.training = True
        
        self.params = {}
        
        if params:
            self.dL_dout = torch.zeros((self.b_size, self.o_size))
        
        self.inputs = torch.zeros((self.b_size, self.i_size))
        self.out = torch.zeros((self.b_size, self.o_size))
        
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
    def __init__(self, input_size, output_size, batch_size):
        super().__init__(input_size, output_size, batch_size, params=False)
    
    def forward(self, x):
        self.inputs[:] = x
        torch.maximum(x, 0, out=self.out)
        return self.out
    
    def backward_p1(self, dL_dout):
        dout_din = torch.where(self.inputs>0, 1.0, 0.0)
        return torch.bmm(dL_dout, dout_din)

    def backward_p2(self):
        pass
    
class BatchNorm2D(Layer):
    def __init__(self, input_size, batch_size, epsilon=1e-3, momentum = 0.99):
        super().__init__(input_size, input_size, batch_size, params=True)
        self.mu = torch.zeros(1, input_size)
        self.var = torch.ones(1, input_size)
        
        self.eps = epsilon
        self.it_call = 0
        self.momentum = momentum
        
        self.params = {
            "beta": torch.zeros(1, input_size),
            "gamma": torch.ones(1, input_size)
            } 
        
        self.grads = {
            "beta": torch.zeros(1, input_size),
            "gamma": torch.zeros(1, input_size)
            } 
    
    def forward(self, x):
        self.it_call += 1
        self.inputs[:] = x
        if self.training:
            b_mu = torch.mean(x, dim=0).unsqueeze(0)
            b_var = torch.var(x, dim=0).unsqueeze(0)
            
            x_norm = (x-b_mu)/torch.sqrt(b_var + self.eps)
            self.out[:] = self.params["gamma"]*x_norm + self.params["beta"]
            
            self.mu = b_mu * (self.momentum/self.it_call) + \
                            self.mu * (1 - (self.momentum/self.it_call))
            
            self.var = b_var * (self.momentum/self.it_call) + \
                        self.var * (1 - (self.momentum/self.it_call))
        
        else:
            x_norm = (x-self.mu)/torch.sqrt(self.var + self.epsilon)
            self.out[:] = self.params["gamma"]*x_norm + self.params["beta"]

        return self.out
    
    def backward_p1(self, dL_dout):
        ...
        