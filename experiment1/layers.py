import torch
import numpy as np

class BaseDense:
    def __init__(self, input_size, output_size, batch_size):
        self.i_size = input_size
        self.o_size = output_size
        self.b_size = batch_size

        self.params = {
            "w": torch.randn((self.i_size, self.o_size))*0.1,
            "b": torch.zeros((self.o_size))
            }
        

        self.inputs = torch.zeros((self.b_size, self.i_size))
        self.out = torch.zeros((self.b_size, self.o_size))

        self.grads = {
            "w": torch.zeros((self.i_size, self.o_size)),
            "b": torch.zeros((self.o_size))
            }
    
    def forward(self, x):
        self.inputs[:] = x 
        x = torch.mm(x, self.params["w"])
        torch.add(x, self.params["b"], out=self.out)
        return self.out
    
    def backward(self, grads):
        self.grads["b"][:] = torch.mean(grads, axis=0)
        self.grads["w"][:] = torch.sum(
            torch.einsum('bij,bjk->bik', 
                         self.inputs.view(self.b_size, self.i_size, 1), 
                         grads.view(self.b_size, 1, 
                                    self.o_size)),
            dim=0
        )
        self.grads["w"] /= self.b_size
        return torch.mm(grads, self.params["w"].T)

    def __call__(self, x):
        return self.forward(x)
    

class CustomBackDense:
    def __init__(self, input_size, output_size, batch_size):
        self.i_size = input_size
        self.o_size = output_size
        self.b_size = batch_size

        self.params = {
            "w": torch.randn((self.i_size, self.o_size))*0.1,
            "b": torch.zeros((self.o_size))
            }
        

        self.inputs = torch.zeros((self.b_size, self.i_size))
        self.out = torch.zeros((self.b_size, self.o_size))

        self.grads = {
            "w": torch.zeros((self.i_size, self.o_size)),
            "b": torch.zeros((self.o_size))
            }
    
    def forward(self, x):
        self.inputs[:] = x 
        x = torch.mm(x, self.params["w"])
        torch.add(x, self.params["b"], out=self.out)
        return self.out
    
    def backward(self, dL_dzout, dzout_dx, dout__dx):
        dx_dout = torch.inverse(dout__dx)
        