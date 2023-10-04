import torch
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, lr, model):
        self.lr = lr
        self.model = model
        self.update_fn()

    @abstractmethod
    def update_fn(self):
        pass
    

class SGD(Optimizer):
    def __init__(self, lr, model):
        super().__init__(lr, model)

    def update_fn(self):
        def _update_fn(_self):
            if not hasattr(_self, "params"):
                return
            for k in _self.params.keys():
                _self.params[k] -= self.lr*_self.grads[k]
        for layer in self.model.layers:
            layer.update = _update_fn


class Adam(Optimizer):
    def __init__(self, lr, model, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        for layer in model.layers:
            layer.t = 0
            base_moments = {}
            for k in layer.params.keys():
                base_moments[k] = torch.zeros_like(layer.params[k])
            layer.m = base_moments
            layer.v = base_moments
        super().__init__(lr, model)
    
    def update_fn(self):
        def _update_fn(_self):
            if not hasattr(_self, "params"):
                return
            _self.t += 1
            for k in _self.params.keys():
                _self.m[k] = self.beta1*_self.m[k] + (1-self.beta1)*_self.grads[k]
                _self.v[k] = self.beta2*_self.v[k] + (1-self.beta2)*(_self.grads[k]**2)
                m_hat = _self.m[k]/(1-self.beta1**_self.t)
                v_hat = _self.v[k]/(1-self.beta2**_self.t)
                _self.params[k] -= self.lr*m_hat/(torch.sqrt(v_hat) + self.epsilon)
        for layer in self.model.layers:
            layer.update = _update_fn
            
            
        
        

