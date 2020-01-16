from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .binary import convert_binary_elementwise


@tensorrt_converter('torch.div')
@tensorrt_converter('torch.Tensor.__div__')  # py2
@tensorrt_converter('torch.Tensor.__idiv__')  # py2
@tensorrt_converter('torch.Tensor.__truediv__')  # py3
@tensorrt_converter('torch.Tensor.__itruediv__')  # py3
def convert_div(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.DIV)


@tensorrt_converter('torch.Tensor.__rdiv__')  # py2
@tensorrt_converter('torch.Tensor.__rtruediv__')  # py3
def convert_rdiv(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.DIV, flip=True)


@tensorrt_converter('torch.Tensor.__floordiv__')  # py2
def convert_div(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.FLOOR_DIV)


@tensorrt_converter('torch.Tensor.__rfloordiv__')  # py3
def convert_rdiv(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.FLOOR_DIV, flip=True)


class Div(torch.nn.Module):
    def __init__(self):
        super(Div, self).__init__()

    def forward(self, x, y):
        return x / y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_basic():
    return Div()


class IDiv(torch.nn.Module):
    def __init__(self):
        super(IDiv, self).__init__()

    def forward(self, x, y):
        x /= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_idiv():
    return IDiv()


class TorchDiv(torch.nn.Module):
    def __init__(self):
        super(TorchDiv, self).__init__()

    def forward(self, x, y):
        return torch.div(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_div_torchdiv():
    return TorchDiv()


class RDivInt(torch.nn.Module):
    def __init__(self):
        super(RDivInt, self).__init__()

    def forward(self, x):
        return 100 / x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rdiv_int():
    return RDivInt()


class RDivFloat(torch.nn.Module):
    def __init__(self):
        super(RDivFloat, self).__init__()

    def forward(self, x):
        return 100.0 / x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rdiv_float():
    return RDivFloat()
