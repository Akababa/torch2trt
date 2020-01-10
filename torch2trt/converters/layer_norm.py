from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


# TODO debug this
@tensorrt_converter('torch.layer_norm')
@tensorrt_converter('torch.nn.functional.layer_norm')
def convert_layer_norm(ctx):
    # TODO Optimize this
    input = get_arg(ctx, 'input', pos=0, default=None)
    normalized_shape = get_arg(ctx, 'normalized_shape', pos=1, default=None)
    weight = get_arg(ctx, 'weight', pos=2, default=None)
    bias = get_arg(ctx, 'bias', pos=3, default=None)
    eps = get_arg(ctx, 'eps', pos=4, default=1e-05)
    # elementwise_affine = get_arg(ctx, 'elementwise_affine', pos=20, default=1e-05)
    output = ctx.method_return
    # print(input.shape, weight.shape, bias.shape)
    input_trt, weight_trt, bias_trt = trt_(ctx.network, input, weight, bias)

    keep_dims = True
    input_ndim = input.ndim
    reduce_axes = torch_dim_to_trt_axes(tuple(range(- len(normalized_shape), 0)), ndim=input_ndim)

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
