import torch
import threading



class BaseModel:
    def __init__(self, *args):
        self.layers = [*args]
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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
            if hasattr(layer, "inputs"):
                layer.inputs = layer.inputs.to(device)
            if hasattr(layer, "out"):
                layer.out = layer.out.to(device)
        

class CustomBackModel:
    def __init__(self, *args):
        self.layers = [*args]
        self.streams = []
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, dL_dzout):
        if "cpu" in dL_dzout.device.type:
            grads = torch.eye(dL_dzout.shape[-1]).to(dL_dzout.device)
            threads = []
            for i, layer in enumerate(self.layers[::-1]):
                t = threading.Thread(target=layer.backward, args=[dL_dzout, grads])
                t.start()
                threads.append(t)
                if i != (len(self.layers)-1):
                    grads = torch.mm(grads, layer.get_jac())
            for thread in threads:
                thread.join()
                        
                
            
        elif "cuda" in dL_dzout.device.type:
            grads = torch.eye(dL_dzout.shape[-1]).to(dL_dzout.device)
            if not self.streams:
                for i in range(len(self.layers)):
                    self.streams.append(torch.cuda.Stream())
            for i, layer in enumerate(self.layers[::-1]):
                with torch.cuda.stream(self.streams[i]):
                    layer.backward(dL_dzout, grads)
                if i != (len(self.streams)-1):
                    grads = torch.mm(grads, layer.get_jac())
            
            for s in self.streams:
                s.synchronize()
                
        
        """
        grads = [torch.eye(dL_dzout.shape[-1]).to(dL_dzout.device)]
        
        for layer in self.layers[:0:-1]:
            grads.append(torch.mm(grads[-1], layer.get_jac()))
        
        fn = lambda layer_idx: self.layers[layer_idx].backward(dL_dzout, grads[-(layer_idx+1)])
        #torch.vmap(fn)(torch.arange(len(self.layers)))
        if "cpu" in dL_dzout.device.type:
            threads = []
            for i in range(len(self.layers)):
                t = threading.Thread(target=fn, args=[i])
                t.start()
                threads.append(t)
            
            for thread in threads:
                thread.join()
        
        elif "cuda" in dL_dzout.device.type:
            if not self.streams:
                for i in range(len(self.layers)):
                    self.streams.append(torch.cuda.Stream())
            else:
                for i in range(len(self.layers)):
                    with torch.cuda.stream(self.streams[i]):
                        fn(i)
                for stream in self.streams:
                    stream.synchronize()
        """    
    
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
