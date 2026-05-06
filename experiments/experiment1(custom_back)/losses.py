import torch
from torch.autograd.functional import jacobian
from functools import partial

def MSE(logits, labels):
    return torch.mean((labels - logits)**2,dim=1)

def MSE_back(logits, labels):
    ... # return jacobian(partial(MSE, labels=labels), logits)

def CCE(logits, labels):
    return - torch.sum(labels * torch.log(logits), dim=1)

def CCE_back(logits, labels):
    return  -1/logits * labels

def softmax_CCE(logits, labels):
    return CCE(torch.softmax(logits,dim=1), labels)


def softmax_CCE_back(logits, labels):
    return torch.softmax(logits,dim=1) - labels
