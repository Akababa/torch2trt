from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule
from .binary import convert_binary


@tensorrt_converter('torch.max')
@tensorrt_converter('torch.Tensor.max')
def convert_max(ctx):
    convert_binary(ctx, trt.ElementWiseOperation.MAX)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_reduce_dim1():
    return UnaryModule(lambda x: torch.max(x, 1)[0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_reduce_dim22():
    return UnaryModule(lambda x: torch.max(x, 2)[0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_max_reduce_dim1_keepdim():
    return UnaryModule(lambda x: torch.max(x, 1, keepdim=True)[0])


class MaxElementwise(torch.nn.Module):
    def forward(self, x, y):
        return torch.max(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1,)])  # broadcast
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3), (1, 3, 3)])  # broadcast
def test_max_elementwise():
    return MaxElementwise()
