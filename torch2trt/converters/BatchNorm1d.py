from ..conversion_context import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.BatchNorm1d.forward')
def convert_BatchNorm2d(ctx: ConversionContext):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = ctx.get_trt_tensor(input)
    output = ctx.method_return

    scale = module.weight.detach().cpu().numpy() / np.sqrt(module.running_var.detach().cpu().numpy() + module.eps)
    bias = module.bias.detach().cpu().numpy() - module.running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)

    # reshape to 2D
    layer = ctx.network.add_shuffle(input_trt)

    if len(input.shape) == 2:
        reshape_dims = (input.shape[1], 1, 1)
    else:
        reshape_dims = (input.shape[1], input.shape[2], 1)
    if ctx.has_implicit_batch():
        reshape_dims = (input.shape[0],) + reshape_dims
    layer.reshape_dims = reshape_dims

    layer = ctx.network.add_scale(layer.get_output(0), trt.ScaleMode.CHANNEL, bias, scale, power)

    # reshape back to 1D
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple(output.shape[ctx.nonbatch_dim:])

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def test_BatchNorm1d_basic():
    return torch.nn.BatchNorm1d(10)
