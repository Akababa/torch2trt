from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.matmul')
@tensorrt_converter('torch.Tensor.matmul')
def convert_matmul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt = trt_(ctx.network, input_a)
    input_b_trt = trt_(ctx.network, input_b)
    output = ctx.method_return
    layer = ctx.network.add_matrix_multiply(input_a_trt, input_b_trt)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.addmm')
@tensorrt_converter('torch.Tensor.addmm')
def convert_addmm(ctx):
    input = get_arg(ctx, "input", 0, None)
    mat1 = get_arg(ctx, "mat1", 1, None)
    mat2 = get_arg(ctx, "mat2", 2, None)
    output = ctx.method_return
    input_trt = trt_(ctx.network, input)
    mat1_trt = trt_(ctx.network, mat1)
    mat2_trt = trt_(ctx.network, mat2)
    m1m2 = ctx.network.add_matrix_multiply(mat1_trt, mat2_trt).get_output(0)
    layer = ctx.network.add_elementwise(m1m2, input_trt, trt.ElementWiseOperation.SUM)
    output._trt = layer.get_output(0)


class TorchMatmul(torch.nn.Module):
    def __init__(self):
        super(TorchMatmul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_torchadd():
    return TorchMatmul()
