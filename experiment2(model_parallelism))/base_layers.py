import torch
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_size, output_size, batch_size):
        self.i_size = input_size
        self.o_size = output_size
        self.b_size = batch_size
        
        self.params = {}
        
        self.inputs = torch.zeros((self.b_size, self.i_size))
        self.out = torch.zeros((self.b_size, self.o_size))
        
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
    
    def forward(self, x):
        self.inputs[:] = x
        torch.maximum(x, 0, out=self.out)
        return self.out
    
    def backward(self, dL_dout):
        dout_din = torch.where(self.inputs>0, 1.0, 0.0)
        return torch.bmm(dL_dout, dout_din)
