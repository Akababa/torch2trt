from ..conversion_context import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.softmax')
def convert_softmax(ctx: ConversionContext):
    input = ctx.method_args[0]
    # input_trt = ctx.get_trt_tensor(input)  # should be a variable
    output = ctx.method_return

    # get dims from args or kwargs

    layer = ctx.network.add_softmax(input=input._trt)
    layer.axes = ctx.get_trt_axes()

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module():
    return torch.nn.Softmax(1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module_dim2():
    return torch.nn.Softmax(2)
