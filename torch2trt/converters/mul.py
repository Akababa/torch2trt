from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .binary import convert_binary_elementwise


@tensorrt_converter('torch.mul')
@tensorrt_converter('torch.Tensor.__imul__')
@tensorrt_converter('torch.Tensor.__mul__')
@tensorrt_converter('torch.Tensor.__rmul__')
def convert_mul(ctx):
    convert_binary_elementwise(ctx, trt.ElementWiseOperation.PROD)


class Mul(torch.nn.Module):
    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x, y):
        return x * y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_basic():
    return Mul()


class IMul(torch.nn.Module):
    def __init__(self):
        super(IMul, self).__init__()

    def forward(self, x, y):
        x *= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_imul():
    return IMul()


class TorchMul(torch.nn.Module):
    def __init__(self):
        super(TorchMul, self).__init__()

    def forward(self, x, y):
        return torch.mul(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_torchmul():
    return TorchMul()


class RMulInt(torch.nn.Module):
    def __init__(self):
        super(RMulInt, self).__init__()

    def forward(self, x):
        return 10 * x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_int():
    return RMulInt()


class RMulFloat(torch.nn.Module):
    def __init__(self):
        super(RMulFloat, self).__init__()

    def forward(self, x):
        return 10.0 * x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_float():
    return RMulFloat()
