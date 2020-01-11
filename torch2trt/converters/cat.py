from ..conversion_context import *


@tensorrt_converter('torch.cat')
def convert_cat(ctx: ConversionContext):
    inputs = ctx.method_args[0]

    # dim = ctx.get_arg("dim", 1, default=1)

    output = ctx.method_return
    trt_inputs = [ctx.get_trt_tensor(i) for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = ctx.get_trt_dim(pos=1, default=1)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.stack')
def convert_cat(ctx: ConversionContext):
    inputs = ctx.method_args[0]

    dim = ctx.get_arg("dim", 1, default=1)
    output = ctx.method_return

    axis = dim if dim >= 0 else output.ndim() + dim
    rshape = list(output.shape[1:]) if ctx.input_has_implicit_batch() else list(output.shape)
    rshape[axis] = 1

    trt_inputs = [ctx.get_trt_tensor(i) for i in inputs]
    trt_inputs_unsqueezed = []
    for trt_in in trt_inputs:
        shuf_l = ctx.network.add_shuffle(trt_in)
        shuf_l.reshape_dims = rshape
        trt_inputs_unsqueezed.append(shuf_l.get_output(0))

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = axis
    output._trt = layer.get_output(0)
