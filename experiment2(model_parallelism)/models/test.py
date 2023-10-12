class BaseModel:
    def __init__(self, *args):
        self.layers = [*args]
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for layer in self.layers:
            #print(type(layer), " ", x.shape)
            x = layer(x)
            #print(x)
        return x
        
    def backward(self, grads):
        for layer in self.layers[::-1]:
            #print(type(layer), " ", grads.shape)
            grads = layer.backward(grads)
    
    def update(self):
        for layer in self.layers:
            layer.update(layer)

    def to(self, device):
        for layer in self.layers:
            if hasattr(layer, "params"):
                for k in layer.params.keys():
                    layer.params[k] = layer.params[k].to(device)
                    layer.grads[k] = layer.grads[k].to(device)
            if hasattr(layer, "inputs"):
                layer.inputs = layer.inputs.to(device)
            if hasattr(layer, "out"):
                layer.out = layer.out.to(device)