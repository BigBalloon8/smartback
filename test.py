import torch


def conv_test(x):
    k1 = torch.ones(1,1,3,3)
    x = torch.nn.functional.conv2d(x, k1, padding="same")
    x = torch.nn.functional.conv2d(x, k1, padding="same")
    x = torch.nn.functional.conv2d(x, k1, padding="same")
    x = torch.nn.functional.conv2d(x, k1, padding="same")
    return torch.sum(x)

torch.manual_seed(0)
img1 = torch.randn((1,1,5,5))
img2 = torch.randn((1,1,5,5))

#print(torch.autograd.functional.jacobian(conv_test, img1))
#print(torch.autograd.functional.jacobian(conv_test, img2))
 
def fn(x):
    return x / sum(x)

