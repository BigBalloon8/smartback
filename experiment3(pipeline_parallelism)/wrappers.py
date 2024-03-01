from functools import wraps
from typing import Any

import torch.distributed as dist
import torch

class cleanup_act(object):
    def __init__(self, *args):
        self.args = args
    def __call__(self, func):
        @wraps(func)
        def wrapper(_self, *args, **kwargs):
            out = func(_self, *args, **kwargs)
            if _self.multi_stage:
                for arg in _self.acts:
                    setattr(_self, arg, [])
            return out
        return wrapper

def multi_stage_wrapper(func):
    # designed to wrap backward_p1
    @wraps(func)
    def wrapper(self, *args, step=0, **kwargs):
        if not self.multi_stage: # or dist.get_rank() == 0:
            out = func(self, *args, step, **kwargs)
            self.backward_p2(step=step)
            return out
        else:
            return func(self, *args, **kwargs)
    return wrapper

class expose_params(object):
    def __init__(self, p_and_g: dict={}, acts:list=[]):
        self.p_and_g = p_and_g 
        self.acts = acts
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            _self = args[0]
            _self.params = {}
            _self.grads = {}
            _self.acts = []
            for p, g in self.p_and_g.items():
                if not hasattr(_self, p) or getattr(_self, p) is None:
                    continue
                _self.params[p] = getattr(_self, p)
                _self.grads[p] = getattr(_self, g)
            _self.acts = self.acts
            return out

        return wrapper

class nvtx_wrapper:
    def __init__(self, name: str):
        self.name = name
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            return out
        return wrapper
        
