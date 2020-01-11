from ..conversion_context import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.mean')
@tensorrt_converter('torch.Tensor.mean')
def convert_mean(ctx: ConversionContext):
    input = ctx.method_args[0]
    input_trt = ctx.get_trt_tensor(input)
    output = ctx.method_return

    dim = ctx.get_trt_dim()

    # get whether to keep dimensions
    keepdim = ctx.get_arg('keepdim', pos=2, default=False)

    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, ctx.get_trt_axes(dim), keepdim)
    output._trt = layer.get_output(0)


class Mean(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(self.dim, self.keepdim)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_channel():
    return Mean(1, False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_tuple():
    return Mean((1, 2), False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_mean_keepdim():
    return Mean(1, True)
