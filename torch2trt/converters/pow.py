from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .binary import convert_binary_elementwise

@tensorrt_converter('torch.pow')
@tensorrt_converter('torch.Tensor.__ipow__')
@tensorrt_converter('torch.Tensor.__pow__')
def convert_pow(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.POW)

    
@tensorrt_converter('torch.Tensor.__rpow__')
def convert_pow(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.POW, flip=True)
    

class Pow(torch.nn.Module):
    def __init__(self):
        super(Pow, self).__init__()

    def forward(self, x, y):
        return x ** y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_pow_basic():
    return Pow()


# __ipow__ not yet impl in torch
# class IPow(torch.nn.Module):
#     def __init__(self):
#         super(IPow, self).__init__()

#     def forward(self, x, y):
#         x **= y
#         return x


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
# def test_pow_ipow():
#     return IPow()


class TorchPow(torch.nn.Module):
    def __init__(self):
        super(TorchPow, self).__init__()

    def forward(self, x, y):
        return torch.pow(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_torch_pow():
    return TorchPow()


class RpowInt(torch.nn.Module):
    def __init__(self):
        super(RpowInt, self).__init__()

    def forward(self, x):
        return 2 ** x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rpow_int():
    return RpowInt()


class RpowFloat(torch.nn.Module):
    def __init__(self):
        super(RpowFloat, self).__init__()

    def forward(self, x):
        return 2.0 ** x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rpow_float():
    return RpowFloat()