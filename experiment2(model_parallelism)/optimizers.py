from abc import ABC, abstractmethod

import torch

import layers
import models

class Optimizer(ABC):
    def __init__(self, model:models.Model, lr:float):
        self.model = model
        self.lr = lr
        self.update_fn()

    @abstractmethod
    def update_fn(self):
        pass
    
    def _recursive_set_attr(self, layer: layers.Layer, attr: tuple):
        if layer.p_to_g:
            setattr(layer, *attr)

        for item in dir(layer):
            if isinstance(getattr(layer, item), layers.Layer):
                self._recursive_set_attr(getattr(layer, item), attr)
            elif isinstance(getattr(layer, item), list):
                for sub_layer in getattr(layer, item):
                    if isinstance(sub_layer, layers.Layer):
                        self._recursive_set_attr(sub_layer, attr)
            elif isinstance(getattr(layer, item), dict):
                for sub_layer in getattr(layer, item).values():
                    if isinstance(sub_layer, layers.Layer):
                        self._recursive_set_attr(sub_layer, attr)
    
    def _recursive_set_empty_opt_states(self, layer: layers.Layer, opt_creation_fn:callable, *args):
        if layer.params:
            for arg in args:
                opt_state = {}
                for k in layer.params.keys():
                    opt_state.update({k: opt_creation_fn(layer.params[k])})
                setattr(layer, arg, opt_state)

        for item in dir(layer):
            if isinstance(getattr(layer, item), layers.Layer):
                self._recursive_set_empty_opt_states(getattr(layer, item), opt_creation_fn, *args)
            elif isinstance(getattr(layer, item), list):
                for sub_layer in getattr(layer, item):
                    if isinstance(sub_layer, layers.Layer):
                        self._recursive_set_empty_opt_states(sub_layer, opt_creation_fn, *args)
            elif isinstance(getattr(layer, item), dict):
                for sub_layer in getattr(layer, item).values():
                    if isinstance(sub_layer, layers.Layer):
                        self._recursive_set_empty_opt_states(sub_layer, opt_creation_fn, *args)
                        

class SGD(Optimizer):
    def __init__(self, lr:float, model:models.Model):
        super().__init__(lr, model)

    def update_fn(self):
        def _update_fn(_self: layers.Layer):
            for k in _self.params.keys():
                _self.params[k] -= self.lr*_self.grads[k]
        for layer in self.model.layers:
            self._recursive_set_attr(layer, ("update", _update_fn))


class Adam(Optimizer):
    def __init__(self, lr, model:models.Model, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        for layer in model.layers:
            self._recursive_set_attr(layer, ("t", 0))
            self._recursive_set_empty_opt_states(layer, torch.zeros_like, "m", "v")
        super().__init__(lr, model)
    
    def update_fn(self):
        def _update_fn(_self: layers.Layer):
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
            self._recursive_set_attr(layer, ("update", _update_fn))
            
