from abc import ABC, abstractmethod

import torch

class Loss(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, y_hat, y):
        pass

    @abstractmethod
    def backward(self):
        pass

class CrossEntropyLoss(Loss):
    def __init__(self, ):
        super().__init__()

    def forward(self, y_hat, y):
        self.logits = torch.softmax(y_hat, dim=-1)
        self.y = y
        return torch.nn.functional.cross_entropy(self.logits, y)

    def backward(self):
        return self.logits - self.y

