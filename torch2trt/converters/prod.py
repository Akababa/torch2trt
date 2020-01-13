from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule
from .binary import convert_binary_reduce


@tensorrt_converter('torch.prod')
@tensorrt_converter('torch.Tensor.prod')
def convert_prod(ctx: ConversionContext):
    convert_binary_reduce(ctx, trt.ReduceOperation.PROD)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_prod_reduce_all():
    return UnaryModule(lambda x: torch.prod(x))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_prod_reduce_dim1():
    return UnaryModule(lambda x: torch.prod(x, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_prod_reduce_dim22():
    return UnaryModule(lambda x: torch.prod(x, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def test_prod_reduce_dim1_keepdim():
    return UnaryModule(lambda x: torch.prod(x, 1, keepdim=True))
