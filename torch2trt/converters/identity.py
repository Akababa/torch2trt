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
def convert_to(ctx: ConversionContext):
    input_trt = ctx.get_arg("input", pos=0, to_trt=True)
    dtype = ctx.get_arg("dtype", 1, None)
    if isinstance(dtype, torch.dtype):
        output_trt = ctx.convert_dtype_to(input_trt, torch_dtype_to_trt(dtype))
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
