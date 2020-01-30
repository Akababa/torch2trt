from ..conversion_context import *


@tensorrt_converter('torch.Tensor.contiguous')
@tensorrt_converter('torch.nn.functional.dropout')
@tensorrt_converter('torch.nn.functional.dropout2d')
@tensorrt_converter('torch.nn.functional.dropout3d')
def convert_identity(ctx):
    input = ctx.method_args[0]
    input_trt = ctx.get_trt_one(input)
    output = ctx.method_return
    output._trt = input_trt


@tensorrt_converter('torch.Tensor.to')
def convert_to(ctx):
    input_trt = ctx.get_arg("input", pos=0, to_trt=True)
    dtype = ctx.get_arg("dtype", 1, None)
    if isinstance(dtype, torch.dtype):
        trt_dtype = torch_dtype_to_trt(dtype)
        layer = ctx.network.add_identity(input_trt)
        # layer.precision = trt_dtype
        layer.set_output_dtype(0, trt_dtype)
        output_trt = layer.get_output(0)
    else:
        output_trt = input_trt

    output = ctx.method_return
    output._trt = output_trt


@tensorrt_converter('torch.nn.Dropout.forward')
@tensorrt_converter('torch.nn.Dropout2d.forward')
@tensorrt_converter('torch.nn.Dropout3d.forward')
def convert_Identity(ctx):
    input = ctx.method_args[1]
    input_trt = ctx.get_trt_one(input)
    output = ctx.method_return
    output._trt = input_trt
