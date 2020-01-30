from ..conversion_context import *
from torch2trt.module_test import add_module_test


def _make_linear(ctx: ConversionContext, input_trt, weight_torch, bias_torch):
    ndims_input = len(input_trt.shape)
    if bias_torch is not None:
        bias = ctx.get_trt_one(bias_torch, weight=True)
    else:
        bias = None

    weight = ctx.get_trt_one(weight_torch, weight=True)

    # reshape to ...xNx1x1
    input_trt_n11 = ctx.reshape_to(input_trt, (0,) * ndims_input + (1, 1))

    # add fully connected
    fc_out_trt = ctx.network.add_fully_connected(
        input=input_trt_n11,
        num_outputs=weight_torch.shape[-2],
        kernel=weight,
        bias=bias).get_output(0)

    # reshape back to N
    return ctx.reshape_to(fc_out_trt, (0,) * ndims_input)


@tensorrt_converter('torch.nn.Linear.forward')
def convert_Linear(ctx: ConversionContext):
    module = ctx.method_args[0]
    input_trt = ctx.get_arg("input", 1, to_trt=True)
    output = ctx.method_return
    output._trt = _make_linear(ctx, input_trt, module.weight, module.bias)


@tensorrt_converter('torch.nn.functional.linear')
def convert_linear(ctx: ConversionContext):
    input_trt = ctx.get_arg("input", 0, to_trt=True)
    weight_torch = ctx.get_arg("weight", 1, to_trt=False)
    bias_torch = ctx.get_arg("bias", 2, default=None, to_trt=False)
    output = ctx.method_return
    output._trt = _make_linear(ctx, input_trt, weight_torch, bias_torch)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_basic():
    return torch.nn.Linear(10, 5)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_no_bias():
    return torch.nn.Linear(10, 5, bias=False)
