from ..conversion_context import *


@tensorrt_converter('torch.where')
def convert_where(ctx: ConversionContext):
    cond = ctx.get_arg("condition", 0, to_trt=True)
    cond = ctx.convert_dtype_to(cond, trt.bool)
    x = ctx.get_arg("x", 1, to_trt=True)
    y = ctx.get_arg("y", 2, to_trt=True)
    x, y = ctx.broadcast_together(x, y)
    cond, x, y = ctx.broadcast_together(cond, x, y, enforce_same_types=False)
    output_trt = ctx.network.add_select(cond, x, y).get_output(0)
    output = ctx.method_return
    output._trt = output_trt
