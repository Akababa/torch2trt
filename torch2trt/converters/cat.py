from torch2trt.torch2trt import *


@tensorrt_converter('torch.cat')
def convert_cat(ctx: ConversionContext):
    inputs = ctx.method_args[0]

    dim = get_arg(ctx, "dim", 1, None)
    if dim < 0:
        dim = inputs[0].dim() - dim
    assert dim >= 1

    output = ctx.method_return
    trt_inputs = [trt_(ctx.network, i) for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim - 1
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.stack')
def convert_cat(ctx: ConversionContext):
    inputs = ctx.method_args[0]

    dim = get_arg(ctx, "dim", 1, default=1)
    if dim < 0:
        dim = inputs[0].dim() + 1 - dim
    assert dim >= 1

    output = ctx.method_return
    rshape = list(output.shape[1:])
    rshape[dim - 1] = 1

    trt_inputs = [trt_(ctx.network, i) for i in inputs]
    trt_inputs_unsqueezed = []
    for trt_in in trt_inputs:
        shuf_l = ctx.network.add_shuffle(trt_in)
        shuf_l.reshape_dims = rshape
        trt_inputs_unsqueezed.append(shuf_l.get_output(0))

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim - 1
    output._trt = layer.get_output(0)
