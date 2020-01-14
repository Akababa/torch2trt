from ..conversion_context import *


def convert_binary(ctx: ConversionContext, trt_op: trt.ElementWiseOperation):
    assert isinstance(trt_op, trt.ElementWiseOperation)
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        convert_binary_elementwise(ctx, trt_op)
    else:
        convert_binary_reduce(ctx, trt_op)


def convert_binary_elementwise(ctx: ConversionContext, trt_op: trt.ElementWiseOperation, flip=False):
    input_a_trt = ctx.get_arg("input", 0, to_trt=True)
    input_b_trt = ctx.get_arg("other", 1, to_trt=True)
    if flip:
        input_a_trt, input_b_trt = input_b_trt, input_a_trt
    input_a_trt, input_b_trt = ctx.broadcast_together(input_a_trt, input_b_trt)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt_op)
    output._trt = layer.get_output(0)


def convert_binary_reduce(ctx: ConversionContext, trt_op: trt.ElementWiseOperation):
    input_trt = ctx.get_arg("input", 0, to_trt=True)
    dim = ctx.get_trt_dim("dim", 1, default="_all", ndims=len(input_trt.shape))
    keepdim = ctx.get_arg('keepdim', pos=2, default=False)
    output_val = ctx.method_return  # [0]
    # output_idx = ctx.method_return[1]  # Hmmmm..
    layer = ctx.network.add_reduce(input_trt, trt_op, ctx.get_trt_axes(dim, ndims=None), keepdim)
    output_val._trt = layer.get_output(0)
