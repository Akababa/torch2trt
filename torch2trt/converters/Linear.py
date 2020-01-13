from ..conversion_context import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.Linear.forward')
def convert_Linear(ctx: ConversionContext):
    module = ctx.method_args[0]
    input_trt = ctx.get_arg("input", 1, to_trt=True)
    output = ctx.method_return

    # reshape to ...xNx1x1
    input_trt_n11 = ctx.reshape_to(input_trt, tuple(input_trt.shape) + (1, 1))

    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    # add fully connected
    fc_out_trt = ctx.network.add_fully_connected(
        input=input_trt_n11,
        num_outputs=module.out_features,
        kernel=module.weight.detach().cpu().numpy(),
        bias=bias).get_output(0)

    # reshape back to N
    output._trt = ctx.reshape_to(fc_out_trt, tuple(output.shape[ctx.nonbatch_dim:]))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_basic():
    return torch.nn.Linear(10, 5)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_no_bias():
    return torch.nn.Linear(10, 5, bias=False)
