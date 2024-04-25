from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
from sympy.abc import y

@contextmanager
def nvtx_profile(name):
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(name)
    yield
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

class Loss(ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    @abstractmethod
    def forward(self, y_hat, y):
        pass

    @abstractmethod
    def backward(self):
        pass

class CrossEntropyLoss(Loss):
    def __init__(self, ):
        super().__init__()
        self.logits = []
        self.y = []

    def forward(self, y_hat, y):
        loss, logits = _nlp_loss_op(y_hat, y)
        self.logits.append(logits)
        self.y.append(y)
        #print(self.logits, y)
        return loss
        return torch.sum(-y*torch.log(logits))

    def backward(self):
        return self.logits.pop(0) - self.y.pop(0)

@torch.jit.script
def _nlp_loss_op(logits, y):
    y_hat = torch.softmax(logits, dim=-1)
    return torch.sum(-y*torch.log(y_hat)), y_hat

class NLPCrossEntropyLoss(Loss):
    def __init__(self, ):
        super().__init__()
        self.y_hat = []
        self.y = []

    def forward(self, logits, y):
        y = torch.nn.functional.one_hot(y, logits.shape[-1])    
        loss , y_hat = _nlp_loss_op(logits, y)
        self.y_hat.append(y_hat)
        self.y.append(y)
        return loss

    def backward(self):
        y = self.y.pop(0)
        y_hat = self.y_hat.pop(0)
        return y_hat - y