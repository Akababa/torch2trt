from ..conversion_context import *
from torch2trt.module_test import add_module_test
from .view import remove_dim, insert_dim


def slice_to_trt(ctx, dim_size, dim_slice):
    start = 0 if dim_slice.start is None else dim_slice.start
    stop = dim_size if dim_slice.stop is None else dim_slice.stop
    stride = 1 if dim_slice.step is None else dim_slice.step
    assert isinstance(stride, int), "dynamic stride not supported"
    assert all(isinstance(s, (int, trt.ITensor)) for s in (start, stop, stride))

    if dim_slice.stop is None and isinstance(start, int) and start < 0:
        if isinstance(stride, int):
            size = (-start - 1) // stride + 1
        else:
            raise ValueError
    elif isinstance(stop, trt.ITensor):
        assert stride == 1, "only stride=1 supported with dynamic slices"
        if start == 0:
            size = stop
        else:
            size = ctx.network.add_elementwise(
                stop, ctx.get_trt_one(start), trt.ElementWiseOperation.SUB).get_output(0)
    elif isinstance(start, trt.ITensor):
        assert stride == 1
        size = ctx.network.add_elementwise(
            ctx.get_trt_one(stop), start, trt.ElementWiseOperation.SUB).get_output(0)
    else:
        assert all(isinstance(s, int) for s in (start, stop, stride))
        size = (stop - start - 1) // stride + 1
    return start, size, stride


def gather_one(ctx, input_trt, idx, axis):
    idx_trt = ctx.get_trt_one(idx)
    return ctx.network.add_gather(input_trt, idx_trt, axis).get_output(0)


USE_GATHER = False  # Gathering is slower


# TODO gather for list and individual indices
# TODO but gather is slower!! TEST and revert this back to slice
# TODO negative dim shape
@tensorrt_converter('torch.Tensor.__getitem__')
def convert_tensor_getitem(ctx: ConversionContext):
    input_trt = ctx.get_trt_one(ctx.method_args[0])
    slices = ctx.method_args[1]
    if isinstance(slices, (int, slice, torch.Tensor)):
        slices = (slices,)
    if isinstance(slices, list):
        raise NotImplementedError("List in getitem not supported")

    ndims = len(input_trt.shape)

    # Step 1 - Replace ellipsis with expanded slices
    num_ellipsis = ndims - len([s for s in slices if s not in [None, Ellipsis]])

    def to_trt_keeping_constant(t):
        if isinstance(t, int) or t is None:
            return t
        elif isinstance(t, torch.Tensor):
            return ctx.get_trt_one(t)
        else:
            raise ValueError

    # standardize slices
    inserted_axes = []
    new_slices = []
    for s in slices:
        if s == Ellipsis:
            while num_ellipsis > 0:
                new_slices.append(slice(None, None, None))
                num_ellipsis -= 1
        elif isinstance(s, slice):
            new_slices.append(slice(*map(to_trt_keeping_constant, (s.start, s.stop, s.step))))
        elif isinstance(s, int):  # Keep the ints for now so we can infer removed dims later on
            new_slices.append(s)
        elif isinstance(s, torch.Tensor):
            new_slices.append(ctx.get_trt_one(s))
        elif s is None:
            inserted_axes.append(len(new_slices))
        else:
            raise ValueError(f"unsupported type {type(s)} in __getitem__")

    # fill missing slices at end
    while len(new_slices) < ndims:
        new_slices.append(slice(None, None, None))

    assert len(new_slices) == ndims
    # print("new_slices:",new_slices)
    # print("input shape:", input.shape)
    # print("input_trt shape:", input_trt.shape)

    # add gather layers
    if USE_GATHER:
        for axis in reversed(range(len(new_slices))):
            if not isinstance(new_slices[axis], slice):
                input_trt = gather_one(ctx, input_trt, new_slices[axis], axis)
                new_slices.pop(axis)

    # Step 3 - Add slice layer
    input_trt_shape = ctx.get_shape_tuple(input_trt) if -1 in input_trt.shape else input_trt.shape
    starts, sizes, strides = [], [], []
    removed_axes = []
    real_slice = False
    for i, s in enumerate(new_slices):
        if s != slice(None, None, None):
            real_slice = True
        if isinstance(s, slice):
            start, size, stride = slice_to_trt(ctx, input_trt_shape[i], s)
            starts.append(start)
            sizes.append(size)
            strides.append(stride)
        elif isinstance(s, (int, trt.ITensor)):
            assert not USE_GATHER
            removed_axes.append(i)
            starts.append(s)
            sizes.append(1)
            strides.append(1)
        else:
            raise ValueError("Invalid slice")

    # print("starts,sizes,strides:", starts, sizes, strides)
    assert len(starts) == len(sizes) == len(strides) == len(input_trt.shape)
    if real_slice:
        output_trt = ctx.slice_tensor(input_trt, starts, sizes, strides)
    else:
        output_trt = input_trt

    if len(removed_axes) > 0:
        output_trt = remove_dim(ctx, output_trt, removed_axes)

    if len(inserted_axes) > 0:
        output_trt = insert_dim(ctx, output_trt, inserted_axes)

    # Step 4 - remove int axes
    output = ctx.method_return
    output._trt = output_trt


class LambdaModule(torch.nn.Module):
    def __init__(self, fn):
        super(LambdaModule, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_tensor_getitem_1d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided():
    return LambdaModule(lambda x: x[:, ::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_offset():
    return LambdaModule(lambda x: x[:, 1::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_range():
    return LambdaModule(lambda x: x[:, 1:3:2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim():
    return LambdaModule(lambda x: x[:, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim_ellipsis():
    return LambdaModule(lambda x: x[:, None, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_dim():
    return LambdaModule(lambda x: x[:, ..., None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_2dim():
    return LambdaModule(lambda x: x[:, ..., None, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_weird_combo():
    return LambdaModule(lambda x: x[:, 0:3:4, None, None, 1, ...])
