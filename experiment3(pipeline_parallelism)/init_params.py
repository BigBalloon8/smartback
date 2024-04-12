import torch
from torch.nn import init
import math

def dense_params(in_features, out_features, bias = True, factory_kwargs={}):
    w = torch.empty(in_features, out_features, **factory_kwargs)
    init.kaiming_uniform_(w, a=math.sqrt(5))
    if bias:
        b = torch.empty(out_features, **factory_kwargs)
        fan_in, _ = init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(b, -bound, bound)
    else:
        b = None
    return w, b

def embedding_params(num_embeddings, embedding_dim, factory_kwargs={}):
    w = torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
    init.normal_(w)
    return w

def conv2d_params(in_channels, out_channels, kernel_size, bias = True, factory_kwargs={}):
    w = torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1], **factory_kwargs)
    init.kaiming_uniform_(w, a=math.sqrt(5))
    if bias:
        b = torch.empty(out_channels, **factory_kwargs)
        fan_in, _ = init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(b, -bound, bound)
    else:
        b = None
    return w, b

def mamba_conv1d_params(in_channels, kernal_size, bias = True, factory_kwargs={}):
    w = torch.empty(in_channels, 1, kernal_size, **factory_kwargs)
    init.kaiming_uniform_(w, a=math.sqrt(5))
    if bias:
        b = torch.empty(in_channels, **factory_kwargs)
        fan_in, _ = init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(b, -bound, bound)
    else:
        b = None
    return w, b