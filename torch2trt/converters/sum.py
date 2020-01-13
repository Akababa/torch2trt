from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule
from .binary import convert_binary_reduce

@tensorrt_converter('torch.sum')
@tensorrt_converter('torch.Tensor.sum')
def convert_sum(ctx: ConversionContext):
    convert_binary_reduce(ctx, trt.ReduceOperation.SUM)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_all():
    return UnaryModule(lambda x: torch.sum(x))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_dim1():
    return UnaryModule(lambda x: torch.sum(x, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_dim22():
    return UnaryModule(lambda x: torch.sum(x, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_sum_reduce_dim1_keepdim():
    return UnaryModule(lambda x: torch.sum(x, 1, keepdim=True))
