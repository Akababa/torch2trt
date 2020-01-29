from ..conversion_context import *


def _gelu_helper(ctx, x):
    ndim = len(x.shape)

    def const_helper(val):
        return ctx.get_trt_one(np.full((1,) * ndim, val))

    f_3 = const_helper(3.0)
    x3coeff = const_helper(0.044715)
    sqrt2overpi = const_helper(0.7978846)
    f_1 = const_helper(1.0)
    f_05 = const_helper(.5)

    x3 = ctx.network.add_elementwise(x, f_3, trt.ElementWiseOperation.POW).get_output(0)
    cx3 = ctx.network.add_elementwise(x3, x3coeff, trt.ElementWiseOperation.PROD).get_output(0)
    xpluscx3 = ctx.network.add_elementwise(x, cx3, trt.ElementWiseOperation.SUM).get_output(0)
    cc_xpluscx3 = ctx.network.add_elementwise(xpluscx3, sqrt2overpi, trt.ElementWiseOperation.PROD).get_output(0)
    tanh_stuff = ctx.network.add_activation(cc_xpluscx3, trt.ActivationType.TANH).get_output(0)
    oneplustanh = ctx.network.add_elementwise(tanh_stuff, f_1, trt.ElementWiseOperation.SUM).get_output(0)
    point5 = ctx.network.add_elementwise(oneplustanh, f_05, trt.ElementWiseOperation.PROD).get_output(0)
    final_output = ctx.network.add_elementwise(point5, x, trt.ElementWiseOperation.PROD).get_output(0)
    
    return final_output


@tensorrt_converter('torch.nn.functional.gelu')
@tensorrt_converter('torch._C._nn.gelu')
def convert_gelu(ctx: ConversionContext):
    x = ctx.get_arg('input', pos=0, to_trt=True)

    output = ctx.method_return
    output._trt = _gelu_helper(ctx, x)


@tensorrt_converter('torch.nn.GELU.forward')
def convert_gelu(ctx: ConversionContext):
    x = ctx.get_arg('input', pos=1, to_trt=True)

    output = ctx.method_return
    output._trt = _gelu_helper(ctx, x)
