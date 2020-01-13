from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .binary import convert_binary_elementwise

@tensorrt_converter('torch.add')
@tensorrt_converter('torch.Tensor.__iadd__')
@tensorrt_converter('torch.Tensor.__add__')
@tensorrt_converter('torch.Tensor.__radd__')
def convert_add(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.SUM)
    

class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_basic():
    return Add()


class IAdd(torch.nn.Module):
    def __init__(self):
        super(IAdd, self).__init__()

    def forward(self, x, y):
        x += y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_iadd():
    return IAdd()


class TorchAdd(torch.nn.Module):
    def __init__(self):
        super(TorchAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_torchadd():
    return TorchAdd()


class RAddInt(torch.nn.Module):
    def __init__(self):
        super(RAddInt, self).__init__()

    def forward(self, x):
        return 1 + x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_radd_int():
    return RAddInt()


class RAddFloat(torch.nn.Module):
    def __init__(self):
        super(RAddFloat, self).__init__()

    def forward(self, x):
        return 1.0 + x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_radd_float():
    return RAddFloat()