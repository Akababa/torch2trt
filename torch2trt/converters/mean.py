from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .binary import convert_binary_reduce


@tensorrt_converter('torch.mean')
@tensorrt_converter('torch.Tensor.mean')
def convert_mean(ctx: ConversionContext):
    convert_binary_reduce(ctx, trt.ReduceOperation.AVG)


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
