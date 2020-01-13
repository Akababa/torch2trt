from ..conversion_context import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.transpose')
@tensorrt_converter('torch.Tensor.transpose')
def convert_transpose(ctx: ConversionContext):
    input_trt = ctx.get_arg("input", 0, to_trt=True)
    # permutation -1 because TRT does not include batch dim
    permutation = list(range(len(input_trt.shape)))
    dim0 = ctx.get_trt_dim(name="dim0", pos=1, ndims=len(input_trt.shape))
    dim1 = ctx.get_trt_dim(name="dim1", pos=2, ndims=len(input_trt.shape))

    permutation[dim0] = dim1
    permutation[dim1] = dim0
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(permutation)
    output = ctx.method_return
    output._trt = layer.get_output(0)


class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1).contiguous()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_transpose_12():
    return Transpose(1, 2)
