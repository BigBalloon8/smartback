from typing import Any


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

    def to(self, device):
        for layer in self.layers:
            for k in layer.params.keys():
                layer.params[k] = layer.params[k].to(device)
                layer.grads[k] = layer.grads[k].to(device)
            layer.inputs = layer.inputs.to(device)
            layer.out = layer.out.to(device)
        

