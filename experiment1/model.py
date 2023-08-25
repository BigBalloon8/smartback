from typing import Any
import ivy

class BaseModel:
    def __init__(self, *args):
        self.layers = [*args]
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            pass
        return x
        
    def backward(self, grads):
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
    
    def update(self):
        for layer in self.layers:
            layer.update(layer)

        
        

