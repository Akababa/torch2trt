from ..conversion_context import *
from torch2trt.module_test import add_module_test


# ASSUME EXPLICIT BATCH MODE to make things easier for now
def insert_dim(ctx, trt_tensor, new_dim):
    ndims = len(trt_tensor.shape)
    new_shape = list(trt_tensor.shape)
    new_shape.insert(new_dim, 1)

    if new_shape.count(-1) > 1:
        layer = ctx.network.add_shuffle(trt_tensor)
        layer.reshape_dims = [0] * ndims + [1]
        perm2 = list(range(ndims))
        perm2.insert(new_dim, ndims)
        layer.second_transpose = perm2
        return layer.get_output(0)
    else:
        return ctx.reshape_to(trt_tensor, new_shape)


@tensorrt_converter('torch.Tensor.reshape')
@tensorrt_converter('torch.Tensor.view')
def convert_view(ctx: ConversionContext):
    input_trt = ctx.get_arg(None, pos=0, to_trt=True)
    new_shape = ctx.method_args[1:]

    output = ctx.method_return
    output._trt = ctx.reshape_to(input_trt, new_shape)


# @tensorrt_converter('torch.Tensor.squeeze')
@tensorrt_converter('torch.Tensor.unsqueeze')
def convert_squeeze(ctx: ConversionContext):
    input_trt = ctx.get_arg("input", 0, to_trt=True)
    ndims = len(input_trt.shape)
    new_dim = ctx.get_trt_dim(pos=1, ndims=ndims)

    output = ctx.method_return
    output._trt = insert_dim(ctx, input_trt, new_dim)


# @tensorrt_converter('torch.flatten')


class View(torch.nn.Module):
    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_1d():
    return View(1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_2d():
    return View(1, 1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_3d():
    return View(1, 1, 1, -1)
