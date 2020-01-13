from ..conversion_context import *
from .view import insert_dim

@tensorrt_converter('torch.cat')
def convert_cat(ctx: ConversionContext):
    inputs = ctx.get_arg("tensors", 0, to_trt=False)
    trt_inputs = [ctx.get_trt_one(inp) for inp in inputs]

    layer = ctx.network.add_concatenation(trt_inputs)
    layer.axis = ctx.get_trt_dim(pos=1, default=0, ndims=len(trt_inputs[0].shape))

    output = ctx.method_return
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.stack')
def convert_cat(ctx: ConversionContext):
    inputs = ctx.get_arg("tensors", 0, to_trt=False)
    trt_inputs = [ctx.get_trt_one(inp) for inp in inputs]

    ndims = len(trt_inputs[0].shape)  # before stack
    new_axis = ctx.get_trt_dim(pos=1, default=0, ndims=ndims)

    trt_inputs_unsqueezed = [insert_dim(ctx, tt, new_axis) for tt in trt_inputs]

    layer = ctx.network.add_concatenation(trt_inputs_unsqueezed)
    layer.axis = new_axis
    output = ctx.method_return
    output._trt = layer.get_output(0)
