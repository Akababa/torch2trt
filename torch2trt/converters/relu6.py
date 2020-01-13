from ..conversion_context import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.ReLU6.forward')
def convert_ReLU6(ctx: ConversionContext):
    input_trt = ctx.get_arg("input", 1, to_trt=True)
    const_6_trt = ctx.get_trt_one(6.0)
    output = ctx.method_return

    input_trt, trt_6 = ctx.broadcast_together(input_trt, const_6_trt)

    layer = ctx.network.add_activation(
        input=input_trt, type=trt.ActivationType.RELU)
    layer = ctx.network.add_elementwise(
        layer.get_output(0), trt_6, trt.ElementWiseOperation.MIN)

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_relu6_basic():
    return torch.nn.ReLU6()


@tensorrt_converter('torch.nn.functional.relu6')
def convert_relu6(ctx):
    ctx.method_args = (torch.nn.ReLU6(),) + ctx.method_args
    convert_ReLU6(ctx)
