from ..conversion_context import *
from torch2trt.module_test import add_module_test


def _matmul(ctx, mat1_trt, mat2_trt):
    op1, op2 = trt.MatrixOperation.NONE, trt.MatrixOperation.NONE
    if len(mat1_trt.shape) == len(mat2_trt.shape) - 1:
        op1 = trt.MatrixOperation.VECTOR
    elif len(mat1_trt.shape) - 1 == len(mat2_trt.shape):
        op2 = trt.MatrixOperation.VECTOR
    else:
        assert len(mat1_trt.shape) == len(mat2_trt.shape)

    assert mat1_trt.dtype == mat2_trt.dtype
    m1m2_trt = ctx.network.add_matrix_multiply(mat1_trt, op1, mat2_trt, op2).get_output(0)
    m1m2_trt = ctx.convert_dtype_to(m1m2_trt, mat1_trt.dtype)
    return m1m2_trt


@tensorrt_converter('torch.matmul')
@tensorrt_converter('torch.Tensor.matmul')
def convert_matmul(ctx):
    mat1_trt = ctx.get_arg("input", 0, to_trt=True)
    mat2_trt = ctx.get_arg("other", 1, to_trt=True)
    output = ctx.method_return
    output._trt = _matmul(ctx, mat1_trt, mat2_trt)  # TODO fix this!!!


@tensorrt_converter('torch.addmm')
@tensorrt_converter('torch.Tensor.addmm')
def convert_addmm(ctx: ConversionContext):
    input_trt = ctx.get_arg("input", 0, to_trt=True)
    mat1_trt = ctx.get_arg("mat1", 1, to_trt=True)
    mat2_trt = ctx.get_arg("mat2", 2, to_trt=True)

    m1m2_trt = _matmul(ctx, mat1_trt, mat2_trt)

    m1m2_trt, input_trt = ctx.broadcast_together(m1m2_trt, input_trt)
    layer = ctx.network.add_elementwise(m1m2_trt, input_trt, trt.ElementWiseOperation.SUM)
    output = ctx.method_return
    output._trt = layer.get_output(0)


class TorchMatmul(torch.nn.Module):
    def __init__(self):
        super(TorchMatmul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_torchadd():
    return TorchMatmul()
