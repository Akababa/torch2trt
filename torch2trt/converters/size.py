from ..conversion_context import *


@tensorrt_converter('torch.Tensor.size')
def convert_size(ctx: ConversionContext):
    if ctx.has_implicit_batch():  # this mode has no dynamic shapes
        return
    input_trt = ctx.get_arg("self", 0, to_trt=True)
    trt_dim = ctx.get_trt_dim("_int", 1, None, ndims=len(input_trt.shape))
    output = ctx.method_return

    # If it's a dynamic shape tensor (according to TRT), return a dynamic add_shape,
    # otherwise do nothing (assume it's static)
    if trt_dim is not None:  # Tensor.size(_int), index into the shape
        assert isinstance(output, int)
        if input_trt.shape[trt_dim] == -1:
            trt_shape_tuple = ctx.get_shape_tuple(input_trt)
            output = torch.tensor(output, dtype=torch.int32)
            output._trt = trt_shape_tuple[trt_dim]
    else:  # Tensor.size(), get the full shape
        assert isinstance(output, torch.Size)
        new_output_trt = ctx.get_shape_tuple(input_trt)
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
