from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.layer_norm')
@tensorrt_converter('torch.nn.functional.layer_norm')
def convert_layer_norm(ctx):
    # TODO Optimize this
    input = get_arg(ctx, 'input', pos=0, default=None)
    normalized_shape = get_arg(ctx, 'normalized_shape', pos=1, default=None)
    weight = get_arg(ctx, 'weight', pos=2, default=None)
    bias = get_arg(ctx, 'bias', pos=3, default=None)
    eps = get_arg(ctx, 'eps', pos=1, default=1e-05)
    # elementwise_affine = get_arg(ctx, 'elementwise_affine', pos=2, default=1e-05)
    output = ctx.method_return

    eps_np = np.array([eps], dtype=np.float32)
    keep_dims = True
    input_ndim = input.ndim
    reduce_axes = torch_dim_to_trt_axes(tuple(range(- len(normalized_shape), 0)), ndim=input_ndim)

    # compute mean over spatial
    mean_trt = ctx.network.add_reduce(input._trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)

    # compute variance over spatial (include eps, to reduce layer count)
    delta_trt = ctx.network.add_elementwise(input._trt, mean_trt, trt.ElementWiseOperation.SUB).get_output(0)
    var_trt = ctx.network.add_scale(delta_trt, trt.ScaleMode.UNIFORM, np.zeros_like(eps_np), np.ones_like(eps_np),
                                    2 * np.ones_like(eps_np)).get_output(0)
    var_trt = ctx.network.add_reduce(var_trt, trt.ReduceOperation.AVG, reduce_axes, keep_dims).get_output(0)

    # compute sqrt(var + eps)
    var_trt = ctx.network.add_scale(var_trt, trt.ScaleMode.UNIFORM, eps_np, np.ones_like(eps_np),
                                    0.5 * np.ones_like(eps_np)).get_output(0)

    # compute final result
    result_trt = ctx.network.add_elementwise(delta_trt, var_trt, trt.ElementWiseOperation.DIV).get_output(0)

    # compute affine (if applicable)
    weight_np = weight.detach().cpu().numpy()
    bias_np = bias.detach().cpu().numpy()
    result_trt = ctx.network.add_scale(result_trt, trt.ScaleMode.ELEMENTWISE, bias_np, weight_np, None).get_output(0)

    # result_trt = _add_scale_1d2d3d(ctx.network, result_trt, trt.ScaleMode.ELEMENTWISE, bias_np, weight_np,
    #                                np.ones_like(bias_np))

    output._trt = result_trt
