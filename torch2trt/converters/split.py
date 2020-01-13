from ..conversion_context import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.split')
@tensorrt_converter('torch.Tensor.split')
def convert_split(ctx: ConversionContext):
    input_trt = ctx.get_arg('input', 0, to_trt=True)
    # we don't need to parse split/chunk (arg 1)
    # since we infer size from output tensors
    trt_dim = ctx.get_trt_dim(pos=2, default=0, ndims=len(input_trt.shape))
    # ctx.get_arg('dim', 2, 0)

    outputs = ctx.method_return

    # assert(dim >= 1)

    start = [0] * len(input_trt.shape[ctx.nonbatch_dim:])  # exclude batch
    # TODO: Fix constants
    stride = [1] * len(start)
    offset = 0

    # add slice layers
    for i, output in enumerate(outputs):
        shape = list(output.shape[ctx.nonbatch_dim:])  # exclude batch dim
        start[trt_dim] = offset
        layer = ctx.network.add_slice(input_trt, start, shape, stride)
        output._trt = layer.get_output(0)
        offset = offset + shape[trt_dim]


class TorchSplit(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TorchSplit, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.split(x, *self.args, **self.kwargs)


class TensorSplit(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(TensorSplit, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.split(*self.args, **self.kwargs)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_split_1_1():
    return TorchSplit(1, 1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_split_2_1():
    return TorchSplit(2, 1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_split_3_1():
    return TorchSplit(3, 1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_torch_split_3_2():
    return TorchSplit(3, 2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_tensor_split_3_2():
    return TensorSplit(3, 2)
