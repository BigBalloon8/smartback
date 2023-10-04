import torch

class Dense:
    def __init__(self, in_size, out_size):
        self.w = torch.randn(in_size, out_size, requires_grad=True)
        self.b = torch.zeros(out_size, requires_grad=True)
        self.input = None
        
    def forward(self, x):
        self.input = x
        return torch.mm(x, self.w) + self.b

    def backward_p1(self, dL_dout):
        return torch.autograd.functional.vjp(self.forward, self.input, dL_dout)

if __name__ == "__main__":
    x = torch.randn(1,32)
    dL_dout = torch.randn(1,10)
    model = Dense(32, 10)
    model.forward(x)
    dL_din = model.backward_p1(dL_dout)
    print(dL_din)