from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .binary import convert_binary_elementwise

@tensorrt_converter('torch.sub')
@tensorrt_converter('torch.Tensor.__isub__')
@tensorrt_converter('torch.Tensor.__sub__')
def convert_sub(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.SUB)

    
@tensorrt_converter('torch.Tensor.__rsub__')
def convert_sub(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.SUB, flip=True)
    

class Sub(torch.nn.Module):
    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, x, y):
        return x - y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_sub_basic():
    return Sub()


class ISub(torch.nn.Module):
    def __init__(self):
        super(ISub, self).__init__()

    def forward(self, x, y):
        x -= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_sub_isub():
    return ISub()


class TorchSub(torch.nn.Module):
    def __init__(self):
        super(TorchSub, self).__init__()

    def forward(self, x, y):
        return torch.sub(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_torch_sub():
    return TorchSub()


class RSubInt(torch.nn.Module):
    def __init__(self):
        super(RSubInt, self).__init__()

    def forward(self, x):
        return 1 - x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_int():
    return RSubInt()


class RSubFloat(torch.nn.Module):
    def __init__(self):
        super(RSubFloat, self).__init__()

    def forward(self, x):
        return 1.0 - x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_float():
    return RSubFloat()