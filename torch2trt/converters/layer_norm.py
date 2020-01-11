from ..conversion_context import *
from torch2trt.module_test import add_module_test


# TODO debug this
@tensorrt_converter('torch.layer_norm')
@tensorrt_converter('torch.nn.functional.layer_norm')
def convert_layer_norm(ctx: ConversionContext):
    # TODO Optimize this
    input = ctx.get_arg('input', pos=0, default=None)
    normalized_shape = ctx.get_arg('normalized_shape', pos=1, default=None)
    weight = ctx.get_arg('weight', pos=2, default=None)
    bias = ctx.get_arg('bias', pos=3, default=None)
    eps = ctx.get_arg('eps', pos=4, default=1e-05)
    # elementwise_affine = ctx.get_arg('elementwise_affine', pos=20, default=1e-05)
    output = ctx.method_return
    # print(input.shape, weight.shape, bias.shape)
    input_trt, weight_trt, bias_trt = ctx.get_trt_tensor(input, weight, bias)
    # print(input_trt.shape, weight_trt.shape, bias_trt.shape)

    keep_dims = True
    reduce_axes = ctx.get_trt_axes(torch_dim=tuple(range(-len(normalized_shape), 0)))

    # compute mean over spatial
    mean_trt = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)
    assert len(mean_trt.shape) >= 0

    delta_trt = ctx.network.add_elementwise(input_trt, mean_trt, trt.ElementWiseOperation.SUB).get_output(0)
    # print(delta_trt.shape)
    assert len(delta_trt.shape) >= 0

    # compute variance over spatial (include eps, to reduce layer count)
    var_trt = ctx.network.add_elementwise(delta_trt, delta_trt, trt.ElementWiseOperation.PROD).get_output(0)
    # print(var_trt.shape)
    assert len(var_trt.shape) >= 0

    var_trt = ctx.network.add_reduce(var_trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)
    # print(var_trt.shape)
    assert len(var_trt.shape) >= 0

    eps_np = np.full([1] * len(var_trt.shape), eps, dtype=np.float32)
    eps_trt = ctx.network.add_constant(eps_np.shape, eps_np).get_output(0)
    var_trt = ctx.network.add_elementwise(var_trt, eps_trt, trt.ElementWiseOperation.SUM).get_output(0)
    # print(var_trt.shape)
    assert len(var_trt.shape) >= 0

    # compute sqrt(var + eps)
    var_trt = ctx.network.add_unary(var_trt, trt.UnaryOperation.SQRT).get_output(0)
    print(var_trt.shape)
    assert len(var_trt.shape) >= 0

    # compute final result
    result_trt = ctx.network.add_elementwise(delta_trt, var_trt, trt.ElementWiseOperation.DIV).get_output(0)
    # print(result_trt.shape)
    assert len(result_trt.shape) >= 0

    # print(weight_trt.shape, bias_trt.shape)
    result_trt = ctx.network.add_elementwise(result_trt, weight_trt, trt.ElementWiseOperation.PROD).get_output(0)
    # print(result_trt.shape)
    result_trt = ctx.network.add_elementwise(result_trt, bias_trt, trt.ElementWiseOperation.SUM).get_output(0)
    # print(result_trt.shape)

    output._trt = result_trt
    assert len(result_trt.shape) >= 0
