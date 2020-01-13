from ..conversion_context import *
from torch2trt.module_test import add_module_test


def _add_scale_1d2d3d(ctx, x_trt, mode, offset, scale, power):
    ndim = len(x_trt.shape)

    y_trt = x_trt

    # shape to 2D
    if ndim != 3:
        y_trt = ctx.reshape_to(y_trt, (x_trt.shape[0], x_trt.shape[1], -1))
        # layer = network.add_shuffle(y_trt)
        # layer.reshape_dims = (x_trt.shape[0], x_trt.shape[1], -1)  # NCH -> NCHW
        # y_trt = layer.get_output(0)

    y_trt = ctx.network.add_scale(y_trt, mode, offset, scale, power).get_output(0)

    # shape to original dimension
    if ndim != 3:
        y_trt = ctx.reshape_to(y_trt, tuple(x_trt.shape))
        # layer = network.add_shuffle(layer.get_output(0)) # Bug??
        # layer.reshape_dims = tuple(x_trt.shape)
        # y_trt = layer.get_output(0)

    return y_trt


@tensorrt_converter('torch.instance_norm')
@tensorrt_converter('torch.nn.functional.instance_norm')
def convert_instance_norm(ctx: ConversionContext):
    input = ctx.get_arg('input', pos=0, default=None)
    running_mean = ctx.get_arg('running_mean', pos=1, default=None)
    running_var = ctx.get_arg('running_var', pos=2, default=None)
    weight = ctx.get_arg('weight', pos=3, default=None)
    bias = ctx.get_arg('bias', pos=4, default=None)
    use_input_stats = ctx.get_arg('use_input_stats', pos=5, default=True)
    momentum = ctx.get_arg('momentum', pos=6, default=0.1)
    eps = ctx.get_arg('eps', pos=7, default=1e-05)
    output = ctx.method_return

    # CASE 1 - USING RUNNING STATISTICS
    if not use_input_stats:

        # equivalent to batch norm
        scale = 1.0 / np.sqrt(running_var.detach().cpu().numpy() + eps)
        offset = -running_mean.detach().cpu().numpy() * scale
        power = np.ones_like(scale)

        if weight is not None:
            scale *= weight.detach().cpu().numpy()
            offset += bias.detach().cpu().numpy()

        result_trt = _add_scale_1d2d3d(ctx, input._trt, trt.ScaleMode.CHANNEL, offset, scale, power)

        output._trt = result_trt

    # CASE 2 - USING INPUT STATS
    else:

        eps_np = np.array([eps], dtype=np.float32)
        keep_dims = True
        reduce_axes = ctx.get_trt_axes(tuple(range(2, input.ndim)), None)

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
        if weight is not None:
            weight_np = weight.detach().cpu().numpy()
            bias_np = bias.detach().cpu().numpy()

            result_trt = _add_scale_1d2d3d(ctx, result_trt, trt.ScaleMode.CHANNEL, bias_np, weight_np,
                                           np.ones_like(bias_np))

        output._trt = result_trt


# STATIC

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def test_instance_norm_1d_static():
    return torch.nn.InstanceNorm1d(10, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def test_instance_norm_2d_static():
    return torch.nn.InstanceNorm2d(10, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def test_instance_norm_3d_static():
    return torch.nn.InstanceNorm3d(10, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def test_instance_norm_1d_static_affine():
    return torch.nn.InstanceNorm1d(10, affine=True, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def test_instance_norm_2d_static_affine():
    return torch.nn.InstanceNorm2d(10, affine=True, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def test_instance_norm_3d_static_affine():
    return torch.nn.InstanceNorm3d(10, affine=True, track_running_stats=True)


# DYNAMIC

# @TODO(jwelsh): 1D dynamic test failing
# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
# def test_instance_norm_1d_dynamic():
#     return torch.nn.InstanceNorm1d(10, track_running_stats=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def test_instance_norm_2d_dynamic():
    return torch.nn.InstanceNorm2d(10, track_running_stats=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def test_instance_norm_3d_dynamic():
    return torch.nn.InstanceNorm3d(10, track_running_stats=False)


# @TODO(jwelsh): 1D dynamic test failing
# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
# def test_instance_norm_1d_dynamic_affine():
#     return torch.nn.InstanceNorm1d(10, affine=True, track_running_stats=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def test_instance_norm_2d_dynamic_affine():
    return torch.nn.InstanceNorm2d(10, affine=True, track_running_stats=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def test_instance_norm_3d_dynamic_affine():
    return torch.nn.InstanceNorm3d(10, affine=True, track_running_stats=False)
