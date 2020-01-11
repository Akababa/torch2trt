from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .unary import UnaryModule


@tensorrt_converter('torch.sum')
@tensorrt_converter('torch.Tensor.sum')
def convert_sum(ctx: ConversionContext):
    input = ctx.method_args[0]
    dim = ctx.get_trt_dim(pos=1, default="_all")
    keepdim = ctx.get_arg('keepdim', pos=2, default=False)
    input_trt = ctx.get_trt_tensor(input)
    output = ctx.method_return
    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.SUM, ctx.get_trt_axes(trt_dim=dim), keepdim)
    output._trt = layer.get_output(0)


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
