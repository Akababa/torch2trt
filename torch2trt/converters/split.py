from ..conversion_context import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.split')
@tensorrt_converter('torch.Tensor.split')
def convert_split(ctx: ConversionContext):
    input_trt = ctx.get_arg('input', 0, to_trt=True)
    ndims = len(input_trt.shape)
    # Assumes it splits evenly along the axis
    # assert -1 not in input_trt
    # we don't need to parse split/chunk (arg 1)
    # since we infer size from output tensors
    split_size = ctx.get_arg("split_size_or_sections", 1)
    trt_dim = ctx.get_trt_dim(pos=2, default=0, ndims=len(input_trt.shape))

    assert isinstance(split_size, int), "only constant split size supported"

    outputs = ctx.method_return

    # construct shape Dims
    input_trt_shape = list(input_trt.shape)
    if input_trt_shape[trt_dim] == -1:
        print("Warning: torch.split converter may not work correctly when split axis is dynamic")

    if -1 not in input_trt_shape[:trt_dim] + input_trt_shape[trt_dim + 1:]:
        sizes = input_trt_shape
        sizes[trt_dim] = split_size
    else:  # need dynamic shape
        sizes = list(ctx.get_shape_tuple(input_trt))
        sizes[trt_dim] = split_size

    # add slice layers
    starts = [0] * ndims
    strides = [1] * ndims
    for i, output in enumerate(outputs):
        output._trt = ctx.slice_tensor(input_trt, starts, sizes, strides)
        starts[trt_dim] += split_size


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
