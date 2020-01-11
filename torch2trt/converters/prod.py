from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule


@tensorrt_converter('torch.prod')
@tensorrt_converter('torch.Tensor.prod')
def convert_prod(ctx: ConversionContext):
    input = ctx.method_args[0]
    dim = ctx.get_trt_dim(pos=1, default="_all")
    keepdim = ctx.get_arg('keepdim', pos=2, default=False)
    input_trt = ctx.get_trt_tensor(input)
    output = ctx.method_return
    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.PROD, ctx.get_trt_axes(trt_dim=dim), keepdim)
    output._trt = layer.get_output(0)


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
