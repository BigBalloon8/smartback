import ivy
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, lr, model):
        self.lr = lr
        self.model = model
        for layer in self.model.layers:
            layer.update = self.update_fn

    @abstractmethod
    def update_fn(self):
        pass
    

class SGD(Optimizer):
    def __init__(self, lr, model):
        super().__init__(lr, model)

    def update_fn(self):
        def _update_fn(_self):
            _self.params -= self.lr*_self.grads
        for layer in self.model.layers:
            layer.update = _update_fn


class Adam(Optimizer):
    def __init__(self, lr, model, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        for layer in model.layers:
            layer.t = 0
            layer.m = ivy.zeros_like(layer.params)
            layer.v = ivy.zeros_like(layer.params)
        super().__init__(lr, model)
    
    def update_fn(self):
        def _update_fn(_self):
            _self.t += 1
            _self.m = self.beta1*_self.m + (1-self.beta1)*_self.grads
            _self.v = self.beta2*_self.v + (1-self.beta2)*(_self.grads**2)
            m_hat = _self.m/(1-self.beta1**_self.t)
            v_hat = _self.v/(1-self.beta2**_self.t)
            _self.params -= self.lr*m_hat/(ivy.sqrt(v_hat) + self.epsilon)
        for layer in self.model.layers:
            layer.update = _update_fn
            
            
        
        

