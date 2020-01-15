from ..conversion_context import *


def get_tuple_of_shape(ctx, t: trt.ITensor):
    # ndims = len(t.shape)
    if -1 in t.shape:
        trt_dyn_shape = ctx.network.add_shape(t).get_output(0)
    new_output_trt = []
    # Detect static/dynamic dims and output either python_int/trt_scalar
    for idx, input_dim in enumerate(t.shape):
        if input_dim == -1:  # make it a torch tensor and add ._trt attribute to it
            output_dim_trt = ctx.get_dim_of_shape(trt_dyn_shape, idx)
            new_output_trt.append(output_dim_trt)
        else:
            new_output_trt.append(input_dim)
    return tuple(new_output_trt)


@tensorrt_converter('torch.Tensor.size')
def convert_size(ctx: ConversionContext):
    if ctx.has_implicit_batch():  # this mode has no dynamic shapes
        return
    # input = ctx.method_args[0]  # type: torch.Tensor
    input_trt = ctx.get_arg("self", 0, to_trt=True)
    trt_dim = ctx.get_trt_dim("_int", 1, None, ndims=len(input_trt.shape))
    output = ctx.method_return

    # If it's a dynamic shape tensor (according to TRT), return a dynamic add_shape,
    # otherwise do nothing (assume it's static)
    if trt_dim is not None:  # Tensor.size(_int), index into the shape
        assert isinstance(output, int)
        if input_trt.shape[trt_dim] == -1:
            trt_dyn_shape = ctx.network.add_shape(input_trt).get_output(0)
            output = torch.tensor(output, dtype=torch.int32)
            output._trt = ctx.get_dim_of_shape(trt_dyn_shape, trt_dim)
    else:  # Tensor.size(), get the full shape
        assert isinstance(output, torch.Size)
        new_output_trt = get_tuple_of_shape(ctx, input_trt)
        new_outputs = []
        for output_dim, output_trt_dim in zip(output, new_output_trt):
            if isinstance(output_trt_dim, trt.ITensor):
                output_dim = torch.tensor(output_dim, dtype=torch.int32)  # TODO device
                output_dim._trt = output_trt_dim
            else:
                assert output_dim == output_trt_dim
            new_outputs.append(output_dim)
        output = tuple(new_outputs)

    ctx.method_return = output  # Overwrite the output because we can't put _trt attribute on a python scalar


# Assume no network has tensors with dynamic NUMBER of dimensions
@tensorrt_converter('torch.Tensor.dim', is_real=False)
def dont_warn(ctx):
    pass
