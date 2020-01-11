from ..conversion_context import *


# not included

@tensorrt_converter('torch.Tensor.shape', is_real=False)
def convert_shape(ctx):
    print("Tensor.shape is static, Tensor.size() is dynamic")


@tensorrt_converter('torch.Tensor.size')
def convert_size(ctx):
    print("Tensor.shape is static, Tensor.size() is dynamic")
    input = ctx.method_args[0]
    input_trt = ctx.get_trt_tensor(input)
    output = ctx.method_return
    output._trt = ctx.network.add_shape(input_trt).get_output(0)
    assert len(output._trt.shape) >= 0


@tensorrt_converter('torch.Tensor.dim', is_real=False)
def dont_warn(ctx):
    print("Tensor.ndim is static, Tensor.dim() is static")

# @tensorrt_converter('torch.Tensor.dim')
# def dont_warn(ctx):
#     print("Tensor.ndim is static, Tensor.dim() is dynamic")
#     input = ctx.method_args[0]
#     input_trt = ctx.get_trt_tensor(input)
#     output = ctx.method_return
#     output._trt = ctx.network.add_shape(input_trt).get_output(0)
#     assert len(output._trt.shape) >= 0
