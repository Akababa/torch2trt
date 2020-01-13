from ..conversion_context import *


@tensorrt_converter('torch.Tensor.size')
def convert_size(ctx: ConversionContext):
    if ctx.has_implicit_batch():  # this mode has no dynamic shapes
        return
    input = ctx.method_args[0]  # type: torch.Tensor
    input_trt = ctx.get_trt_tensor(input)
    trt_dim = ctx.get_trt_dim("_int", 1, None)
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
        if -1 in input_trt.shape:
            trt_dyn_shape = ctx.network.add_shape(input_trt).get_output(0)
            new_output = []
            for idx, input_dim, output_dim in zip(range(len(output)), input_trt.shape, output):
                if input_dim == -1:
                    output_dim = torch.tensor(output_dim, dtype=torch.int32)
                    output_dim._trt = ctx.get_dim_of_shape(trt_dyn_shape, idx)
                    new_output.append(output_dim)
                else:
                    assert input_dim == output_dim
                    new_output.append(input_dim)
            output = tuple(new_output)

    ctx.method_return = output


# Assume no network has tensors with dynamic NUMBER of dimensions
@tensorrt_converter('torch.Tensor.dim', is_real=False)
def dont_warn(ctx):
    pass

# @tensorrt_converter('torch.Tensor.dim')
# def dont_warn(ctx):
#     print("Tensor.ndim is static, Tensor.dim() is dynamic")
#     input = ctx.method_args[0]
#     input_trt = ctx.get_trt_tensor(input)
#     output = ctx.method_return
#     output._trt = ctx.network.add_shape(input_trt).get_output(0)
#     assert len(output._trt.shape) >= 0
